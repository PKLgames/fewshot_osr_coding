import torch
import torch.nn as nn
from .module.prototype_dynamic_aggregation import PrototypeDynamicAggregation
from .module.conditional_information_coupling import ConditionalInformationCouplingModule
import math
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
from speechbrain.processing.features import STFT, Filterbank
import torch.nn.functional as F
from torch.nn import NLLLoss
from .module.transformer import OpenSetGenerater,TransformerEncoder
from .module.resnet18_encoder import resnet18


class Backbone(nn.Module):
    def __init__(self,args):
        super(Backbone,self).__init__()
        self.args = args
        self.encoder = resnet18(True,args)
        self.fc = nn.Linear(512,self.args.train_classes, bias=True)
        self.set_module_for_audio()
        
    def forward(self, x):
        x = self.spectrogram_extractor(x)   
        x = self.logmel_extractor(x)    
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        x = x.repeat(1, 3, 1, 1)
        resfeat,_ = self.encoder(x)
        cls_logit = self.fc(resfeat)
        return resfeat, cls_logit
    
    def set_module_for_audio(self):
            center = True
            pad_mode = 'reflect'
            ref = 1.0
            amin = 1e-10
            top_db = None
            # Spectrogram extractor
            self.spectrogram_extractor = Spectrogram(n_fft=self.args.extractor.window_size, hop_length=self.args.extractor.hop_size, 
                win_length=self.args.extractor.window_size, window=self.args.extractor.window, center=center, pad_mode=pad_mode, 
                freeze_parameters=True)

            # Logmel feature extractor
            self.logmel_extractor = LogmelFilterBank(sr=self.args.extractor.sample_rate, n_fft=self.args.extractor.window_size, 
                n_mels=self.args.extractor.mel_bins, fmin=self.args.extractor.fmin, fmax=self.args.extractor.fmax, ref=ref, amin=amin, top_db=top_db, 
                freeze_parameters=True)

            # Spec augmenter
            self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
                freq_drop_width=8, freq_stripes_num=2)
            self.bn0 = nn.BatchNorm2d(self.args.extractor.mel_bins)

            # speechbrain tools 
            self.compute_STFT = STFT(sample_rate=self.args.extractor.sample_rate, 
                                win_length=int(self.args.extractor.window_size / self.args.extractor.sample_rate * 1000), 
                                hop_length=int(self.args.extractor.hop_size / self.args.extractor.sample_rate * 1000), 
                                n_fft=self.args.extractor.window_size)
            self.compute_fbanks = Filterbank(n_mels=self.args.extractor.mel_bins)

class My_Net(nn.Module):
    def __init__(self, args=None,mode=None, use_ciam=True, use_pam=True, use_npm=True):
        super().__init__()
        self.args = args
        self.mode = mode
        # Ablation flags: toggle core components on/off
        self.use_ciam = use_ciam
        self.use_pam = use_pam
        self.use_npm = use_npm
        self.shots = [self.args.train_shot, self.args.train_query_shot]
        self.way = self.args.train_way
        self.resnet = self.args.resnet
        self.metric  = Metric_Cosine()
        self.num_channel = 512
        self.dim = 512 * 52
        self.encoder = resnet18(True,args)
        self.PAM = PrototypeDynamicAggregation(self.num_channel) if use_pam else None
        self.CIAM = ConditionalInformationCouplingModule(512,512,1) if use_ciam else None
        self.NPM = OpenSetGenerater(self.num_channel, n_head=1,agg='mlp') if use_npm else None
        # Fallback for ablation: simple mean pool when PAM is off
        self.gap = nn.AdaptiveAvgPool2d((1, 1)) 

        self.set_module_for_audio()
        self.fc = nn.Linear(self.num_channel,self.args.train_classes, bias=True)
        # self.criterion = NLLLoss()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

    def forward(self,input,labels=None,supp_ids=None,open_ids=None,test=False):
         if self.mode == 'pretrain':
              input,_ = self.encode(input)
              input = self.fc(input)
              return input
         elif self.mode == 'train':
        
            support_data, query_data, suppopen_data, openset_data = input
            (support_label,query_label,supopen_label,openset_label) = labels

            support_feat,s1= self.encode(support_data.cuda())
            q1,query_feat = self.encode(query_data.cuda())
            supopen_feat,so1= self.encode(suppopen_data.cuda())
            # fewshot: openset可能为空(未知类样本不足)，跳过encode
            if openset_data.numel() > 0 and openset_data.dim() >= 2:
                openset_feat,q2 = self.encode(openset_data.cuda())
            else:
                openset_feat = torch.empty(0, query_feat.shape[-1]).cuda()
                q2 = torch.empty(0, query_feat.shape[-1]).cuda()
            #Task 1: dynamically compute number of known classes from query_label
            actual_n_ways = query_label.max().item() + 1
            open_label = actual_n_ways * torch.ones_like(openset_label)
            # fewshot: openset为空时cls_label仅含query部分
            if openset_label.numel() > 0:
                cls_label = torch.cat([query_label, open_label])
            else:
                cls_label = query_label

            if test:
                 loss_cls,loss_fake,prediction = self.task(s1,support_feat,query_feat,q1,openset_feat,support_label.cuda(),cls_label.cuda(),query_label.cuda())
                 return prediction,loss_cls,loss_fake
            loss_cls,loss_fake,prediction = self.task(s1,support_feat,query_feat,q1,openset_feat,support_label.cuda(),cls_label.cuda(),query_label.cuda(),supp_ids.cuda())
            #Task 2: data augmentation with open classes as "known" and query classes as "unknown"
            # Dynamically compute open class count from supopen_label
            actual_n_open_ways = supopen_label.max().item() + 1
            # Map openset_label to contiguous indices [0, actual_n_open_ways-1]
            unique_open = torch.unique(openset_label)
            open_to_idx = {v.item(): i for i, v in enumerate(unique_open)}
            open_query_label = torch.tensor([open_to_idx[l.item()] for l in openset_label],
                                            dtype=torch.long, device=openset_label.device)
            # Query samples are "unknown" in Task 2
            task2_unknown_label = actual_n_open_ways * torch.ones_like(query_label)
            cls_label_aug = torch.cat([open_query_label, task2_unknown_label])
            # Use supopen_label for prototype generation
            loss_cls_aug,loss_aug_fake,_= self.task(so1,supopen_feat,q2,openset_feat,q1,supopen_label.cuda(),cls_label_aug.cuda(),open_query_label,supp_ids.cuda())


            return prediction,(loss_cls+loss_cls_aug,loss_fake+loss_aug_fake)

    def task(self,s1,support_feat,query_feat,q1,openset_feat,support_label,cls_label,query_label,supp_ids=None):
        # CIAM: conditional information coupling (query-conditioned support)
        if self.use_ciam and self.CIAM is not None:
            aug_supp = self.CIAM(s1,query_feat)
        else:
            aug_supp = support_feat

        # PAM: prototype dynamic aggregation (weighted prototype generation)
        if self.use_pam and self.PAM is not None:
            supp_protos = self.PAM(aug_supp,support_label)
        else:
            # Simple mean pooling fallback
            supp_protos = self.gap(aug_supp).squeeze(-1).squeeze(-1)
            out_feats = []
            label = torch.unique(support_label, sorted=False)
            for cls_id in label:
                out_feat = supp_protos[support_label == cls_id]
                out_feat = torch.mean(out_feat, dim=0).unsqueeze(0)
                out_feats.append(out_feat)
            supp_protos = torch.cat(out_feats, dim=0)

        base_weights,base_open_weights = self.get_representation(supp_ids)

        # NPM: open-set prototype generation
        if self.use_npm and self.NPM is not None:
            recip_units, fake_center = self.NPM(supp_protos,base_weights,base_open_weights)
        else:
            # Simple fallback: mean of base open weights as fake center
            fake_center = base_open_weights.mean(dim=0, keepdim=True).unsqueeze(0)  # [1, 1, D]

        cls_protos = torch.cat([supp_protos.unsqueeze(0), fake_center], dim=1)

        query_score = self.metric(cls_protos,q1.unsqueeze(0)).squeeze()
        # fewshot: openset_feat可能为空
        if openset_feat.numel() > 0:
            open_score = self.metric(cls_protos,openset_feat.unsqueeze(0)).squeeze()
            cls_score =torch.cat([query_score.squeeze(),open_score.squeeze()],dim=0)
            loss_cls =F.cross_entropy(cls_score, cls_label)

            # funit_distance = self.metric(recip_units.transpose(0,1),q1.unsqueeze(0)).squeeze()
            # qopen_funit_distance = self.metric(recip_units.transpose(0,1), openset_feat.unsqueeze(0)).squeeze()
            # funit_distance = torch.cat([funit_distance,qopen_funit_distance],dim=0)

            loss_fake = 0.0#fakeunit_compare(funit_distance,self.args.n_ways,cls_label)

            query_score = F.softmax(query_score.detach(), dim=-1).squeeze()
            open_score = F.softmax(open_score.detach(), dim=-1).squeeze()
        else:
            cls_score = query_score.squeeze()
            loss_cls = F.cross_entropy(cls_score, cls_label)
            loss_fake = 0.0
            query_score = F.softmax(query_score.detach(), dim=-1).squeeze()
            open_score = torch.empty(0, query_score.shape[-1] if query_score.dim() > 1 else self.args.train_classes + 1).cuda()
        
        return loss_cls,loss_fake,(query_score,open_score)      

    def encode(self, x):
        x = self.spectrogram_extractor(x)   
        x = self.logmel_extractor(x)    
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        x = x.repeat(1, 3, 1, 1)
        x,x1 = self.encoder(x)
        # if self.mode == 'pretrain':
        #     x = F.adaptive_avg_pool2d(x, 1)
        #     x = x.squeeze(-1).squeeze(-1)
            # x = self.PAM(x)
        return x,x1
        
    def init_representation(self, params):
        if 'RPL_params' in params and 'GCPL_params' in params:
            # Legacy format: RPL/GCPL centers saved explicitly
            params_RPL = params['RPL_params']
            params_GCPL = params['GCPL_params']
            base_open = params_RPL['centers']
            self.weight_base_open = nn.Parameter( base_open * self.args.open_weight_sum_cali, requires_grad=True)
            base = params_GCPL['centers'].view(self.args.train_classes, 512)
            self.weight_base = nn.Parameter(base * self.args.open_weight_sum_cali , requires_grad=True)
        else:
            # New format: derive centers from fc layer weights in state_dict
            state_dict = params.get('params', params.get('feature_params', params))
            fc_weight = state_dict['fc.weight']  # (train_classes, 512)
            self.weight_base = nn.Parameter(fc_weight * self.args.open_weight_sum_cali, requires_grad=True)
            self.weight_base_open = nn.Parameter(-fc_weight * self.args.open_weight_sum_cali, requires_grad=True)

    def get_representation(self, base_ids=None):
        if base_ids is not None and len(base_ids) > 0:
            base_weights = self.weight_base[base_ids,:]   ## bs*54*D
            base_open_weights = self.weight_base_open[base_ids,:]
        else:
            base_weights = self.weight_base.unsqueeze(0)
            base_open_weights = self.weight_base_open.unsqueeze(0)

        return base_weights,base_open_weights
       

    def set_module_for_audio(self):
            center = True
            pad_mode = 'reflect'
            ref = 1.0
            amin = 1e-10
            top_db = None
            # Spectrogram extractor
            self.spectrogram_extractor = Spectrogram(n_fft=self.args.extractor.window_size, hop_length=self.args.extractor.hop_size, 
                win_length=self.args.extractor.window_size, window=self.args.extractor.window, center=center, pad_mode=pad_mode, 
                freeze_parameters=True)

            # Logmel feature extractor
            self.logmel_extractor = LogmelFilterBank(sr=self.args.extractor.sample_rate, n_fft=self.args.extractor.window_size, 
                n_mels=self.args.extractor.mel_bins, fmin=self.args.extractor.fmin, fmax=self.args.extractor.fmax, ref=ref, amin=amin, top_db=top_db, 
                freeze_parameters=True)

            # Spec augmenter
            self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
                freq_drop_width=8, freq_stripes_num=2)
            self.bn0 = nn.BatchNorm2d(self.args.extractor.mel_bins)

            # speechbrain tools 
            self.compute_STFT = STFT(sample_rate=self.args.extractor.sample_rate, 
                                win_length=int(self.args.extractor.window_size / self.args.extractor.sample_rate * 1000), 
                                hop_length=int(self.args.extractor.hop_size / self.args.extractor.sample_rate * 1000), 
                                n_fft=self.args.extractor.window_size)
            self.compute_fbanks = Filterbank(n_mels=self.args.extractor.mel_bins)


class Metric_Cosine(nn.Module):
    def __init__(self, temperature=10.0):
        super(Metric_Cosine, self).__init__()
        self.temp = nn.Parameter(torch.tensor(float(temperature)))

    def forward(self, supp_center, query_feature):
        supp_center = F.normalize(supp_center, dim=-1) # eps=1e-6 default 1e-12
        query_feature = F.normalize(query_feature, dim=-1)
        logits = torch.bmm(query_feature, supp_center.transpose(1,2))
        # logits = torch.cosine_similarity(query_feature, supp_center, dim=-1)
        return logits * self.temp
    


def fakeunit_compare(funit_distance,n_ways,cls_label):
    # cls_label_binary = F.one_hot(cls_label).float()
    cls_label_binary = F.one_hot(cls_label.unsqueeze(0))[:,:,:-1].float().squeeze()
    loss = torch.sum(F.binary_cross_entropy_with_logits(input=funit_distance, target=cls_label_binary))
    return loss   




    