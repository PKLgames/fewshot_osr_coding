#!/usr/local/miniconda3/envs/py312/bin/python

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
from typing import Tuple, Optional, Dict, Any
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import copy
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


from utils.TAU19 import TAUDataset
import yamnet_PT_inference as yamnet_infer
from torch_audioset.yamnet.model import yamnet as torch_yamnet
from Flow_Models import FlowBasedTissue, FlowBasedCell
from loss_function.loss_function_v4 import LossF
from loss_function.loss_function_v4 import LossF

from exam_tool import check_for_anomalies
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

class ModelInterface(nn.Module):
    def __init__(self, num_known_classes: int = 10):
    def __init__(self, num_known_classes: int = 10):
        super().__init__()
        self.num_known_classes = num_known_classes
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def get_config(self) -> Dict[str, Any]:
        return {'num_known_classes': self.num_known_classes}
        return {'num_known_classes': self.num_known_classes}
    
    def save(self, path: str):
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.get_config()
        }, path)
    
    @classmethod
    def load(cls, path: str, device: str = 'cpu'):
        checkpoint = torch.load(path, map_location=device)
        model = cls(**checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        return model


class PolicyNet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(PolicyNet, self).__init__()
        self.in_dim = in_dim
        self.fc1 = nn.Linear(in_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 32)
        self.fc1 = nn.Linear(in_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.drop2 = nn.Dropout(0.4)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(32, out_dim)

    def forward(self, x, temp):
        x = self.drop1(F.leaky_relu(self.bn1(self.fc1(x)), 0.1))
        x = self.drop2(F.leaky_relu(self.bn2(self.fc2(x)), 0.1))
        logits = self.fc3(x)
        hard_mask = F.gumbel_softmax(logits, tau=temp, hard=True, dim=-1)
        return hard_mask


class FinalModel(ModelInterface):
    def __init__(self, 
                 num_unknown_classes: int = 8,
                 num_known_classes: int = 6,
                 reload_feature_model_pretrained: bool = True):
        super().__init__(num_known_classes)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cpudevice = "cpu"
        # 使用VGGish作为特征提取模型
        self.feature_model = VGGish().to(self.device)
        self.num_unknown_classes = num_unknown_classes
        self.num_known_classes = num_known_classes
        self.reload_feature_model_pretrained = reload_feature_model_pretrained

        # 特征维度 - VGGish输出128维
        feature_num = 32
        condition_dim = 0
        self.condition_vector = torch.zeros(condition_dim).to(self.device)

        if reload_feature_model_pretrained:
            path = 'vggish/vggish_pytorch.pth'
            state = torch.load(path, map_location=self.device)
            self.feature_model.load_state_dict(state)
            print(f"已加载VGGish预训练权重 from {path}")

        classifier_cluster_num = 8

        # VGGish输出128维，不是521维
        self.fc1 = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128), 
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64), 
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(64, feature_num),
            nn.BatchNorm1d(feature_num), 
            nn.Tanh()
        )

        self.num_experts = 3
        self.num_experts_cluster = 3
        self.num_experts = 3
        self.num_experts_cluster = 3
        self.classifier_experts = nn.ModuleList([FlowBasedTissue(
                                                input_dim=feature_num,
                                                prototype_num=num_known_classes+1,
                                                prototype_num=num_known_classes+1,
                                                condition_dim=condition_dim,
                                                num_coupling_layers=3,
                                                hidden_dims=[32, 32],
                                                num_coupling_layers=3,
                                                hidden_dims=[32, 32],
                                                use_permutation=True,
                                                permutation_type='fixed',
                                                cluster_num=classifier_cluster_num
                                                cluster_num=classifier_cluster_num
                                                ) for _ in range(self.num_experts)])
        self.clusterer_experts = nn.ModuleList([FlowBasedCell(
                                                input_dim = feature_num,
                                                condition_dim = condition_dim,
                                                num_coupling_layers = 2,
                                                hidden_dims = [32, 32],
                                                num_coupling_layers = 2,
                                                hidden_dims = [32, 32],
                                                use_permutation = True,
                                                permutation_type = 'fixed',
                                                cluster_num = num_unknown_classes
                                                ) for _ in range(self.num_experts_cluster)])
        self.gate1 = PolicyNet(feature_num, self.num_experts)
        self.gate2 = PolicyNet(feature_num, self.num_experts_cluster)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = vggish_infer.waveform_to_log_mel_patches_gpu(x, sample_rate=16000)  # [batch, N, 96, 64] on GPU
        
        feature = self.feature_model(x, to_prob=False)
        feature = self.fc1(feature)
        feature = self.fc2(feature)
        feature = self.fc3(feature)
        feature = self.fc3(feature)
        batch_size = feature.size(0)

        weights1 = self.gate1(feature, 1)
        weights1 = top_k_gating(weights1, k=self.num_experts)

        weights1 = self.gate1(feature, 1)
        weights1 = top_k_gating(weights1, k=self.num_experts)
        
        preds = torch.zeros(batch_size, self.num_known_classes+1, device=self.device)
        preds = torch.zeros(batch_size, self.num_known_classes+1, device=self.device)
        for i in range(self.num_experts):
            expert_output = self.classifier_experts[i](feature, self.condition_vector)
            preds += weights1[:, i].unsqueeze(1) * expert_output
            preds += weights1[:, i].unsqueeze(1) * expert_output

        preds = F.softmax(preds, dim=1)

        masked_feature, _ = get_masked_feature(feature, preds)
        masked_feature, _ = get_masked_feature(feature, preds)
        masked_feature_aug = feature_augment(masked_feature)

        weights2 = self.gate2(feature, 1)
        weights2 = top_k_gating(weights2, k=self.num_experts_cluster)

        weights2 = self.gate2(feature, 1)
        weights2 = top_k_gating(weights2, k=self.num_experts_cluster)

        masked_preds = torch.zeros(batch_size, self.num_unknown_classes, device=self.device)
        full_log_probs = torch.zeros(batch_size, self.num_unknown_classes, device=self.device)
        masked_preds_aug = torch.zeros(batch_size, self.num_unknown_classes, device=self.device)
        full_log_probs_aug = torch.zeros(batch_size, self.num_unknown_classes, device=self.device)
        for i in range(self.num_experts_cluster):
            expert_preds, expert_log_probs = self.clusterer_experts[i](masked_feature, self.condition_vector)
            expert_preds_aug, expert_log_probs_aug = self.clusterer_experts[i](masked_feature_aug, self.condition_vector)
            masked_preds += weights2[:, i].unsqueeze(1) * expert_preds
            full_log_probs += weights2[:, i].unsqueeze(1) * expert_log_probs
            masked_preds_aug += weights2[:, i].unsqueeze(1) * expert_preds_aug
            full_log_probs_aug += weights2[:, i].unsqueeze(1) * expert_log_probs_aug

        masked_preds = F.softmax(masked_preds, dim=1)
        masked_preds_aug = F.softmax(masked_preds_aug, dim=1)
        masked_preds = F.softmax(masked_preds, dim=1)
        masked_preds_aug = F.softmax(masked_preds_aug, dim=1)

        return preds, masked_preds, masked_preds_aug, weights1, weights2
    
    def get_config(self) -> Dict[str, Any]:
        return {
            'num_known_classes': self.num_known_classes,
            'num_unknown_classes': self.num_unknown_classes,
            'reload_feature_model_pretrained': self.reload_feature_model_pretrained
        }


class Trainer:
    def __init__(self,
                model: ModelInterface,
                train_loader: DataLoader,
                calib_loader: DataLoader,
                test_loader: DataLoader,
                train_dataset,
                calib_dataset,
                test_dataset,
                device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.calib_loader = calib_loader
        self.test_loader = test_loader
        self.device = device
        
        self.load_balance_weight = 0.0001
        self.accumulation_steps = 1

        self.num_known_classes = self.model.num_known_classes
        num_unknown_classes = self.model.num_unknown_classes
        
        self.model_criterion = LossF()
        
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=0.0001,
            weight_decay=1e-3,
            amsgrad=True
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=50,
            eta_min=1e-7
        )
        
        self.patience = 15
        self.best_val_loss = float('inf')
        self.early_stop_counter = 0
        
        self.history = {
            'total_train_loss': [],
            'train_loss': [],
            'calib_loss': [],
            'train_acc': [],
            'calib_acc': [],
            'calib_cluster_acc': [],
            'test_loss': [],
            'test_acc': [],
            'test_cluster_acc': [],
            'calib_cluster_ACC': [],
            'calib_cluster_NMI': [],
            'calib_cluster_ARI': [],
            'test_cluster_ACC': [],
            'test_cluster_NMI': [],
            'test_cluster_ARI': []
        }

    def train_epoch(self, epoch: int = 0, total_epochs: int = 30) -> Tuple[float, float]:
        # Handle edge case: empty data loader (e.g., small dataset with drop_last=True)
        if len(self.train_loader) == 0:
            print(f'  Warning: Training loader is empty. Skipping this epoch.')
            return 0.0, 0.0
            
            'test_cluster_acc': [],
            'calib_cluster_ACC': [],
            'calib_cluster_NMI': [],
            'calib_cluster_ARI': [],
            'test_cluster_ACC': [],
            'test_cluster_NMI': [],
            'test_cluster_ARI': []
        }

    def train_epoch(self, epoch: int = 0, total_epochs: int = 30) -> Tuple[float, float]:
        # Handle edge case: empty data loader (e.g., small dataset with drop_last=True)
        if len(self.train_loader) == 0:
            print(f'  Warning: Training loader is empty. Skipping this epoch.')
            return 0.0, 0.0
            
        self.model.train()
        train_running_loss = 0.0
        train_correct = 0
        train_total = 0
        self.optimizer.zero_grad()
        self.optimizer.zero_grad()
        
        for batch_idx, item in enumerate(self.train_loader):
            inputs, targets = item['source_audio'].to(self.device), item['target'].squeeze(1).to(self.device)
            compressed_targets = compress_targets(targets, self.num_known_classes)
            compressed_targets = compressed_targets.argmax(dim=1).long()
            targets = targets.argmax(dim=1).long()

            known_outputs, unknown_outputs, unknown_outputs_aug, weights1, weights2 = self.model(inputs)

            train_loss, _ = self.model_criterion(
                known_outputs, unknown_outputs, unknown_outputs_aug, 
                compressed_targets, weights1, weights2,
                epoch=epoch, total_epochs=total_epochs, num_known_classes=self.num_known_classes
            )
            

            train_loss, _ = self.model_criterion(
                known_outputs, unknown_outputs, unknown_outputs_aug, 
                compressed_targets, weights1, weights2,
                epoch=epoch, total_epochs=total_epochs, num_known_classes=self.num_known_classes
            )
            
            train_loss = train_loss / self.accumulation_steps
            train_loss.backward()
            
            if (batch_idx + 1) % self.accumulation_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            train_running_loss += train_loss.item() * self.accumulation_steps
            _, predicted = known_outputs.max(1)
            train_total += compressed_targets.size(0)
            train_correct += predicted.eq(compressed_targets).sum().item()
            
            if batch_idx % 100 == 99:
                print(f'  Batch: {batch_idx+1}, Train_Loss: {train_loss.item() * self.accumulation_steps:.4f}')
        
        train_epoch_loss = train_running_loss / len(self.train_loader)
        train_epoch_acc = 100. * train_correct / train_total
    
        return train_epoch_loss, train_epoch_acc

    def calib_epoch(self, epoch: int = 0, total_epochs: int = 30) -> Tuple[float, float, float, Dict[str, float]]:
        # Handle edge case: empty data loader (e.g., small dataset with drop_last=True)
        if len(self.calib_loader) == 0:
            print(f'  Warning: Calibration loader is empty. Skipping this epoch.')
            return 0.0, 0.0, 0.0, {'ACC': 0.0, 'NMI': 0.0, 'ARI': 0.0}
            
        self.model.train()
    def calib_epoch(self, epoch: int = 0, total_epochs: int = 30) -> Tuple[float, float, float, Dict[str, float]]:
        # Handle edge case: empty data loader (e.g., small dataset with drop_last=True)
        if len(self.calib_loader) == 0:
            print(f'  Warning: Calibration loader is empty. Skipping this epoch.')
            return 0.0, 0.0, 0.0, {'ACC': 0.0, 'NMI': 0.0, 'ARI': 0.0}
            
        self.model.train()
        calib_running_loss = 0.0
        calib_correct = 0
        calib_total = 0
        
        all_targets = []
        all_compressed_targets = []
        all_known_outputs = []
        all_unknown_outputs = []
        
        all_targets = []
        all_compressed_targets = []
        all_known_outputs = []
        all_unknown_outputs = []
        
        self.optimizer.zero_grad()
        self.optimizer.zero_grad()
        
        for batch_idx, item in enumerate(self.calib_loader):
            inputs, targets = item['source_audio'].to(self.device), item['target'].squeeze(1).to(self.device)
            compressed_targets_onehot = compress_targets(targets, self.num_known_classes)
            compressed_targets_idx = compressed_targets_onehot.argmax(dim=1).long()
            targets_idx = targets.argmax(dim=1).long()

            compressed_targets_onehot = compress_targets(targets, self.num_known_classes)
            compressed_targets_idx = compressed_targets_onehot.argmax(dim=1).long()
            targets_idx = targets.argmax(dim=1).long()

            known_outputs, unknown_outputs, unknown_outputs_aug, weights1, weights2 = self.model(inputs)

            calib_loss, _ = self.model_criterion(
                known_outputs, unknown_outputs, unknown_outputs_aug, 
                compressed_targets_idx, weights1, weights2,
                epoch=epoch, total_epochs=total_epochs, num_known_classes=self.num_known_classes
            )
            

            calib_loss, _ = self.model_criterion(
                known_outputs, unknown_outputs, unknown_outputs_aug, 
                compressed_targets_idx, weights1, weights2,
                epoch=epoch, total_epochs=total_epochs, num_known_classes=self.num_known_classes
            )
            
            calib_loss = calib_loss / self.accumulation_steps
            calib_loss.backward()
            
            if (batch_idx + 1) % self.accumulation_steps == 0 or (batch_idx + 1) == len(self.calib_loader):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            calib_running_loss += calib_loss.item() * self.accumulation_steps
            calib_total += compressed_targets_idx.size(0)
            calib_total += compressed_targets_idx.size(0)
            _, predicted = known_outputs.max(1)
            calib_correct += predicted.eq(compressed_targets_idx).sum().item()
            
            all_targets.append(targets.cpu())
            all_compressed_targets.append(compressed_targets_onehot.cpu())
            all_known_outputs.append(known_outputs.detach().cpu())
            all_unknown_outputs.append(unknown_outputs.detach().cpu())

            calib_correct += predicted.eq(compressed_targets_idx).sum().item()
            
            all_targets.append(targets.cpu())
            all_compressed_targets.append(compressed_targets_onehot.cpu())
            all_known_outputs.append(known_outputs.detach().cpu())
            all_unknown_outputs.append(unknown_outputs.detach().cpu())

            if batch_idx % 100 == 99:
                print(f'  Batch: {batch_idx+1}, Calib_Loss: {calib_loss.item() * self.accumulation_steps:.4f}')
        
        calib_epoch_loss = calib_running_loss / len(self.calib_loader)
        calib_epoch_acc = 100. * calib_correct / calib_total
        
        all_targets = torch.cat(all_targets, dim=0)
        all_compressed_targets = torch.cat(all_compressed_targets, dim=0)
        all_known_outputs = torch.cat(all_known_outputs, dim=0)
        all_unknown_outputs = torch.cat(all_unknown_outputs, dim=0)
        
        clustering_metrics = evaluate_clustering(
            all_targets, all_compressed_targets,
            all_known_outputs, all_unknown_outputs,
            self.num_known_classes, prefix="校准"
        )
        
        calib_epoch_cluster_acc = clustering_metrics['ACC'] * 100.0
        
        return calib_epoch_loss, calib_epoch_acc, calib_epoch_cluster_acc, clustering_metrics
        
        all_targets = torch.cat(all_targets, dim=0)
        all_compressed_targets = torch.cat(all_compressed_targets, dim=0)
        all_known_outputs = torch.cat(all_known_outputs, dim=0)
        all_unknown_outputs = torch.cat(all_unknown_outputs, dim=0)
        
        clustering_metrics = evaluate_clustering(
            all_targets, all_compressed_targets,
            all_known_outputs, all_unknown_outputs,
            self.num_known_classes, prefix="校准"
        )
        
        calib_epoch_cluster_acc = clustering_metrics['ACC'] * 100.0
        
        return calib_epoch_loss, calib_epoch_acc, calib_epoch_cluster_acc, clustering_metrics

    def test(self) -> Tuple[float, float, float, Dict[str, float]]:
        # Handle edge case: empty data loader (e.g., small dataset with drop_last=True)
        if len(self.test_loader) == 0:
            print(f'  Warning: Test loader is empty. Skipping evaluation.')
            return 0.0, 0.0, 0.0, {'ACC': 0.0, 'NMI': 0.0, 'ARI': 0.0}
            
        self.model.eval()
    def test(self) -> Tuple[float, float, float, Dict[str, float]]:
        # Handle edge case: empty data loader (e.g., small dataset with drop_last=True)
        if len(self.test_loader) == 0:
            print(f'  Warning: Test loader is empty. Skipping evaluation.')
            return 0.0, 0.0, 0.0, {'ACC': 0.0, 'NMI': 0.0, 'ARI': 0.0}
            
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        all_targets = []
        all_compressed_targets = []
        all_known_outputs = []
        all_unknown_outputs = []
        
        all_targets = []
        all_compressed_targets = []
        all_known_outputs = []
        all_unknown_outputs = []
        
        with torch.no_grad():
            for batch_idx, item in enumerate(self.test_loader):
                inputs = item['source_audio'].to(self.device, non_blocking=True)
                targets_onehot = item['target'].squeeze(1).to(self.device, non_blocking=True)
                
                compressed_targets_onehot = compress_targets(targets_onehot, self.num_known_classes)
                compressed_targets_idx = compressed_targets_onehot.argmax(dim=1).long()
                targets_idx = targets_onehot.argmax(dim=1).long()

                known_outputs, unknown_outputs, unknown_outputs_aug, weights1, weights2 = self.model(inputs)

                loss, _ = self.model_criterion(known_outputs, unknown_outputs, unknown_outputs_aug, compressed_targets_idx, weights1, weights2)
                
                running_loss += loss.item()
                total += compressed_targets_idx.size(0)
                total += compressed_targets_idx.size(0)
                _, predicted = known_outputs.max(1)
                correct += predicted.eq(compressed_targets_idx).sum().item()
                
                all_targets.append(targets_onehot.cpu())
                all_compressed_targets.append(compressed_targets_onehot.cpu())
                all_known_outputs.append(known_outputs.cpu())
                all_unknown_outputs.append(unknown_outputs.cpu())
                correct += predicted.eq(compressed_targets_idx).sum().item()
                
                all_targets.append(targets_onehot.cpu())
                all_compressed_targets.append(compressed_targets_onehot.cpu())
                all_known_outputs.append(known_outputs.cpu())
                all_unknown_outputs.append(unknown_outputs.cpu())
        
        test_loss = running_loss / len(self.test_loader)
        test_acc = 100. * correct / total
        
        all_targets = torch.cat(all_targets, dim=0)
        all_compressed_targets = torch.cat(all_compressed_targets, dim=0)
        all_known_outputs = torch.cat(all_known_outputs, dim=0)
        all_unknown_outputs = torch.cat(all_unknown_outputs, dim=0)
        
        clustering_metrics = evaluate_clustering(
            all_targets, all_compressed_targets,
            all_known_outputs, all_unknown_outputs,
            self.num_known_classes, prefix="测试"
        )
        
        test_cluster_acc = clustering_metrics['ACC'] * 100.0
        
        return test_loss, test_acc, test_cluster_acc, clustering_metrics
    
    def train_two_stage(self, stage1_epochs: int = 30, stage2_epochs: int = 20):
        print(f'开始两阶段训练，设备: {self.device}')
        print('=' * 70)
        
        total_epochs = stage1_epochs + stage2_epochs
        current_epoch = 0
        
        print('\n' + '=' * 70)
        print('第一阶段：训练分类器（Stage 1 - Classifier Training）')
        print('=' * 70)
        
        self.model_criterion = LossF(
            stage1_lambda_ce=1.5,
            stage1_lambda_binary=1.0,
            stage1_lambda_unk_enc=0.8,
            stage1_lambda_warmup=0.5,
            stage1_lambda_balance=0.1,
            stage1_lambda_expert=0.1,
            stage1_label_smoothing=0.25,
            stage1_focal_gamma=2.0,
            stage1_lambda_conf_penalty=0.0,
            stage2_lambda_ce=0.2,
            stage2_lambda_binary=0.1,
            stage2_lambda_kd=0.0,
            stage2_lambda_ps=2.5,
            stage2_lambda_sharp=1.2,
            stage2_lambda_me_max=1.5,
            stage2_lambda_flow=0.3,
            stage2_lambda_expert=0.05,
            stage2_label_smoothing=0.05,
            stage2_sharp_temp=0.5,
            stage2_lambda_mi=1.2,
            stage2_lambda_conf_penalty=0.0,
        )
        self.model_criterion.set_stage(1)
        
        optimizer_stage1 = optim.AdamW(
            self.model.parameters(), 
            lr=0.0001,
            weight_decay=2e-3,
            amsgrad=True
        )
        
        scheduler_stage1 = optim.lr_scheduler.CosineAnnealingLR(
            optimizer_stage1, 
            T_max=stage1_epochs, 
            eta_min=1e-7
        )
        
        self.best_val_loss = float('inf')
        self.early_stop_counter = 0
        
        for epoch in range(stage1_epochs):
            print(f'\n[Stage 1] Epoch {epoch+1}/{stage1_epochs}')
            print('-' * 50)
            
            train_loss, train_acc = self.train_epoch_with_optimizer(
                epoch=epoch, 
                total_epochs=stage1_epochs, 
                optimizer=optimizer_stage1
            )
            
            calib_loss, calib_acc, calib_cluster_acc, calib_clustering_metrics = self.calib_epoch_with_optimizer(
                epoch=epoch, 
                total_epochs=stage1_epochs,
                optimizer=optimizer_stage1
            )
            
            test_loss, test_acc, test_cluster_acc, test_clustering_metrics = self.test()
            
            scheduler_stage1.step()
            
            self.history['train_loss'].append(train_loss)
            self.history['calib_loss'].append(calib_loss)
            self.history['train_acc'].append(train_acc)
            self.history['calib_acc'].append(calib_acc)
            self.history['calib_cluster_acc'].append(calib_cluster_acc)
            self.history['test_loss'].append(test_loss)
            self.history['test_acc'].append(test_acc)
            self.history['calib_cluster_ACC'].append(calib_clustering_metrics['ACC'])
            self.history['calib_cluster_NMI'].append(calib_clustering_metrics['NMI'])
            self.history['calib_cluster_ARI'].append(calib_clustering_metrics['ARI'])
            self.history['test_cluster_ACC'].append(test_clustering_metrics['ACC'])
            self.history['test_cluster_NMI'].append(test_clustering_metrics['NMI'])
            self.history['test_cluster_ARI'].append(test_clustering_metrics['ARI'])
            
            print(f'[Stage 1] 训练 Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%')
            print(f'[Stage 1] 校准 Loss: {calib_loss:.4f}, Acc: {calib_acc:.2f}%, Cluster Acc: {calib_cluster_acc:.2f}%')
            print(f'[Stage 1] 验证 Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%, Cluster Acc: {test_cluster_acc:.2f}%')
            
            gap = train_acc - test_acc
            print(f'  >> 过拟合差距: {gap:.2f}%')
            
            if test_loss < self.best_val_loss:
                self.best_val_loss = test_loss
                self.early_stop_counter = 0
                self.model.save('experiment/yamnet_C2MoE_2stageN/stage1_best.pth')
                print(f'  [Stage 1] 保存最佳模型 (Val Loss: {test_loss:.4f})')
            else:
                self.early_stop_counter += 1
                print(f'  [Stage 1] 早停计数: {self.early_stop_counter}/{self.patience}')
                print(f'  [Stage 1] 早停计数: {self.early_stop_counter}/{self.patience}')
                if self.early_stop_counter >= self.patience:
                    print(f'\n!!! [Stage 1] 早停触发，进入第二阶段 !!!')
                    print(f'\n!!! [Stage 1] 早停触发，进入第二阶段 !!!')
                    break
            
            current_epoch += 1
        
        print('\n' + '=' * 70)
        print('第二阶段：训练聚类器（Stage 2 - Clustering Training）')
        print('=' * 70)
        
        self.model_criterion.set_stage(2)
        
        if os.path.exists('experiment/yamnet_C2MoE_2stageN/stage1_best.pth'):
            print('加载第一阶段最佳模型...')
            checkpoint = torch.load('experiment/yamnet_C2MoE_2stageN/stage1_best.pth')
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        optimizer_stage2 = optim.AdamW(
            self.model.parameters(), 
            lr=0.00005,
            weight_decay=1e-3,
            amsgrad=True
        )
        
        scheduler_stage2 = optim.lr_scheduler.CosineAnnealingLR(
            optimizer_stage2, 
            T_max=stage2_epochs, 
            eta_min=5e-8
        )
        
        self.best_cluster_acc = test_cluster_acc
        self.early_stop_counter = 0
        
        for epoch in range(stage2_epochs):
            print(f'\n[Stage 2] Epoch {epoch+1}/{stage2_epochs}')
            print('-' * 50)
            
            train_loss, train_acc = self.train_epoch_with_optimizer(
                epoch=epoch, 
                total_epochs=stage2_epochs, 
                optimizer=optimizer_stage2
            )
            
            calib_loss, calib_acc, calib_cluster_acc, calib_clustering_metrics = self.calib_epoch_with_optimizer(
                epoch=epoch, 
                total_epochs=stage2_epochs,
                optimizer=optimizer_stage2
            )
            
            test_loss, test_acc, test_cluster_acc, test_clustering_metrics = self.test()
            
            scheduler_stage2.step()
            
            self.history['train_loss'].append(train_loss)
            self.history['calib_loss'].append(calib_loss)
            self.history['train_acc'].append(train_acc)
            self.history['calib_acc'].append(calib_acc)
            self.history['calib_cluster_acc'].append(calib_cluster_acc)
            self.history['test_loss'].append(test_loss)
            self.history['test_acc'].append(test_acc)
            self.history['calib_cluster_ACC'].append(calib_clustering_metrics['ACC'])
            self.history['calib_cluster_NMI'].append(calib_clustering_metrics['NMI'])
            self.history['calib_cluster_ARI'].append(calib_clustering_metrics['ARI'])
            self.history['test_cluster_ACC'].append(test_clustering_metrics['ACC'])
            self.history['test_cluster_NMI'].append(test_clustering_metrics['NMI'])
            self.history['test_cluster_ARI'].append(test_clustering_metrics['ARI'])
            
            print(f'[Stage 2] 训练 Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%')
            print(f'[Stage 2] 校准 Loss: {calib_loss:.4f}, Acc: {calib_acc:.2f}%, Cluster Acc: {calib_cluster_acc:.2f}%')
            print(f'[Stage 2] 验证 Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%, Cluster Acc: {test_cluster_acc:.2f}%')
            
            if test_cluster_acc > self.best_cluster_acc:
                self.best_cluster_acc = test_cluster_acc
                self.early_stop_counter = 0
                self.model.save('experiment/yamnet_C2MoE_2stageN/stage2_best.pth')
                print(f'  [Stage 2] 保存最佳模型 (Cluster Acc: {test_cluster_acc:.2f}%)')
            else:
                self.early_stop_counter += 1
                print(f'  [Stage 2] 早停计数: {self.early_stop_counter}/{self.patience}')
                if self.early_stop_counter >= self.patience:
                    print(f'\n!!! [Stage 2] 早停触发，停止训练 !!!')
                    break
            
            current_epoch += 1
        
        print('\n' + '=' * 70)
        print(f'两阶段训练完成！总轮次: {current_epoch}')
        print('=' * 70)

    def train_epoch_with_optimizer(self, epoch: int = 0, total_epochs: int = 30, 
                                   optimizer = None) -> Tuple[float, float]:
        if optimizer is None:
            optimizer = self.optimizer
            
        # Handle edge case: empty data loader (e.g., small dataset with drop_last=True)
        if len(self.train_loader) == 0:
            print(f'  Warning: Training loader is empty. Skipping this epoch.')
            return 0.0, 0.0
            
        self.model.train()
        train_running_loss = 0.0
        train_correct = 0
        train_total = 0
        optimizer.zero_grad()
        
        for batch_idx, item in enumerate(self.train_loader):
            inputs, targets = item['source_audio'].to(self.device), item['target'].squeeze(1).to(self.device)
            compressed_targets = compress_targets(targets, self.num_known_classes)
            compressed_targets = compressed_targets.argmax(dim=1).long()
            targets = targets.argmax(dim=1).long()

            known_outputs, unknown_outputs, unknown_outputs_aug, weights1, weights2 = self.model(inputs)

            train_loss, _ = self.model_criterion(
                known_outputs, unknown_outputs, unknown_outputs_aug, 
                compressed_targets, weights1, weights2,
                epoch=epoch, total_epochs=total_epochs, num_known_classes=self.num_known_classes
            )
            
            train_loss = train_loss / self.accumulation_steps
            train_loss.backward()
            
            if (batch_idx + 1) % self.accumulation_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            train_running_loss += train_loss.item() * self.accumulation_steps
            _, predicted = known_outputs.max(1)
            train_total += compressed_targets.size(0)
            train_correct += predicted.eq(compressed_targets).sum().item()
            
            if batch_idx % 100 == 99:
                print(f'  Batch: {batch_idx+1}, Train_Loss: {train_loss.item() * self.accumulation_steps:.4f}')
        
        train_epoch_loss = train_running_loss / len(self.train_loader)
        train_epoch_acc = 100. * train_correct / train_total
        
        return train_epoch_loss, train_epoch_acc

    def calib_epoch_with_optimizer(self, epoch: int = 0, total_epochs: int = 30, 
                                   optimizer = None) -> Tuple[float, float, float, Dict[str, float]]:
        if optimizer is None:
            optimizer = self.optimizer
            
        # Handle edge case: empty data loader (e.g., small dataset with drop_last=True)
        if len(self.calib_loader) == 0:
            print(f'  Warning: Calibration loader is empty. Skipping this epoch.')
            return 0.0, 0.0, 0.0, {'ACC': 0.0, 'NMI': 0.0, 'ARI': 0.0}
            
        self.model.train()
        calib_running_loss = 0.0
        calib_correct = 0
        calib_total = 0
        
        all_targets = []
        all_compressed_targets = []
        all_known_outputs = []
        all_unknown_outputs = []
        
        optimizer.zero_grad()
        
        for batch_idx, item in enumerate(self.calib_loader):
            inputs, targets = item['source_audio'].to(self.device), item['target'].squeeze(1).to(self.device)
            compressed_targets_onehot = compress_targets(targets, self.num_known_classes)
            compressed_targets_idx = compressed_targets_onehot.argmax(dim=1).long()
            targets_idx = targets.argmax(dim=1).long()

            known_outputs, unknown_outputs, unknown_outputs_aug, weights1, weights2 = self.model(inputs)

            calib_loss, _ = self.model_criterion(
                known_outputs, unknown_outputs, unknown_outputs_aug, 
                compressed_targets_idx, weights1, weights2,
                epoch=epoch, total_epochs=total_epochs, num_known_classes=self.num_known_classes
            )
            
            calib_loss = calib_loss / self.accumulation_steps
            calib_loss.backward()
            
            if (batch_idx + 1) % self.accumulation_steps == 0 or (batch_idx + 1) == len(self.calib_loader):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            calib_running_loss += calib_loss.item() * self.accumulation_steps
            calib_total += compressed_targets_idx.size(0)
            _, predicted = known_outputs.max(1)
            calib_correct += predicted.eq(compressed_targets_idx).sum().item()
            
            all_targets.append(targets.cpu())
            all_compressed_targets.append(compressed_targets_onehot.cpu())
            all_known_outputs.append(known_outputs.detach().cpu())
            all_unknown_outputs.append(unknown_outputs.detach().cpu())

            if batch_idx % 100 == 99:
                print(f'  Batch: {batch_idx+1}, Calib_Loss: {calib_loss.item() * self.accumulation_steps:.4f}')
        
        calib_epoch_loss = calib_running_loss / len(self.calib_loader)
        calib_epoch_acc = 100. * calib_correct / calib_total
        
        all_targets = torch.cat(all_targets, dim=0)
        all_compressed_targets = torch.cat(all_compressed_targets, dim=0)
        all_known_outputs = torch.cat(all_known_outputs, dim=0)
        all_unknown_outputs = torch.cat(all_unknown_outputs, dim=0)
        
        clustering_metrics = evaluate_clustering(
            all_targets,
            all_compressed_targets,
            all_known_outputs,
            all_unknown_outputs,
            self.num_known_classes,
            prefix="校准"
        )
        
        calib_epoch_cluster_acc = clustering_metrics['ACC'] * 100.0
        
        return calib_epoch_loss, calib_epoch_acc, calib_epoch_cluster_acc, clustering_metrics

    def plot_history(self, from_epochs: int, to_epochs: int):
        fig1, axes1 = plt.subplots(1, 2, figsize=(12, 4))
        
        axes1[0].plot(self.history['train_loss'], label='Train Loss')
        axes1[0].plot(self.history['calib_loss'], label='Calib Loss')
        axes1[0].plot(self.history['test_loss'], label='Test Loss')
        axes1[0].set_xlabel('Epoch')
        axes1[0].set_ylabel('Loss')
        axes1[0].set_title('Training and Validation Loss')
        axes1[0].legend()
        axes1[0].grid(True)
        
        axes1[1].plot(self.history['train_acc'], label='Train Acc')
        axes1[1].plot(self.history['calib_acc'], label='Calib Acc')
        axes1[1].plot(self.history['calib_cluster_acc'], label='Calib Cluster ACC')
        axes1[1].plot(self.history['test_acc'], label='Test Acc')
        axes1[1].plot(self.history['test_cluster_acc'], label='Test Cluster ACC')
        axes1[1].set_xlabel('Epoch')
        axes1[1].set_ylabel('Accuracy (%)')
        axes1[1].set_title('Training and Validation Accuracy')
        axes1[1].legend()
        axes1[1].grid(True)
        axes1[1].plot(self.history['train_acc'], label='Train Acc')
        axes1[1].plot(self.history['calib_acc'], label='Calib Acc')
        axes1[1].plot(self.history['calib_cluster_acc'], label='Calib Cluster ACC')
        axes1[1].plot(self.history['test_acc'], label='Test Acc')
        axes1[1].plot(self.history['test_cluster_acc'], label='Test Cluster ACC')
        axes1[1].set_xlabel('Epoch')
        axes1[1].set_ylabel('Accuracy (%)')
        axes1[1].set_title('Training and Validation Accuracy')
        axes1[1].legend()
        axes1[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(f'experiment/yamnet_C2MoE_2stageN/yamnet_C2MoE_2stageN_trained_from{str(from_epochs)}to{str(to_epochs)}.png')
        
        fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
        
        axes2[0].plot(self.history['calib_cluster_ACC'], label='Calib ACC', marker='o', markersize=4)
        axes2[0].plot(self.history['test_cluster_ACC'], label='Test ACC', marker='s', markersize=4)
        axes2[0].set_xlabel('Epoch')
        axes2[0].set_ylabel('ACC')
        axes2[0].set_title('Clustering Accuracy (ACC)')
        axes2[0].legend()
        axes2[0].grid(True)
        axes2[0].set_ylim([0, 1])
        
        axes2[1].plot(self.history['calib_cluster_NMI'], label='Calib NMI', marker='o', markersize=4)
        axes2[1].plot(self.history['test_cluster_NMI'], label='Test NMI', marker='s', markersize=4)
        axes2[1].set_xlabel('Epoch')
        axes2[1].set_ylabel('NMI')
        axes2[1].set_title('Normalized Mutual Information (NMI)')
        axes2[1].legend()
        axes2[1].grid(True)
        axes2[1].set_ylim([0, 1])
        
        axes2[2].plot(self.history['calib_cluster_ARI'], label='Calib ARI', marker='o', markersize=4)
        axes2[2].plot(self.history['test_cluster_ARI'], label='Test ARI', marker='s', markersize=4)
        axes2[2].set_xlabel('Epoch')
        axes2[2].set_ylabel('ARI')
        axes2[2].set_title('Adjusted Rand Index (ARI)')
        axes2[2].legend()
        axes2[2].grid(True)
        axes2[2].set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(f'experiment/yamnet_C2MoE_2stageN/yamnet_C2MoE_2stageN_clustering_metrics_from{str(from_epochs)}to{str(to_epochs)}.png')
        
        print(f'\n图表已保存:')
        print(f'  1. 损失和准确率: experiment/yamnet_C2MoE_2stageN/yamnet_C2MoE_2stageN_trained_from{str(from_epochs)}to{str(to_epochs)}.png')
        print(f'  2. 聚类指标: experiment/yamnet_C2MoE_2stageN/yamnet_C2MoE_2stageN_clustering_metrics_from{str(from_epochs)}to{str(to_epochs)}.png')


def get_masked_feature(feature: torch.Tensor, preds: torch.Tensor) -> torch.Tensor:
    unknown_probs = preds[:, -1]
    mask_f = unknown_probs.unsqueeze(1)
    mask_f = mask_f.expand_as(feature)
    mask_p = unknown_probs.unsqueeze(1).expand_as(preds)
    masked_feature = feature * mask_f
    masked_preds = preds * mask_p
    return masked_feature, masked_preds


def feature_augment(
    features: torch.Tensor,
    p_apply: float = 1.0,
    p_apply: float = 1.0,
    aug_list: list = None) -> torch.Tensor:
    if aug_list is None:
        aug_list = [
            'none',
            'noise',
            'time_mask',
            'scale',
            'invert'
        ]

    batch_size, feat_dim = features.shape
    device = features.device
    augmented = features.clone()

    for i in range(batch_size):
        if random.random() > p_apply:
            continue
            continue

        sample = features[i]
        sample = features[i]
        aug_type = random.choice(aug_list)

        if aug_type == 'none':
            augmented[i] = sample
        elif aug_type == 'noise':
            noise = torch.randn_like(sample, device=device) * 0.05
            noise = torch.randn_like(sample, device=device) * 0.05
            augmented[i] = sample + noise
        elif aug_type == 'time_mask':
            mask_len = random.randint(1, min(6, feat_dim // 3))
            start = random.randint(0, feat_dim - mask_len)
            aug_sample = sample.clone()
            aug_sample[start:start + mask_len] = 0.0
            augmented[i] = aug_sample
        elif aug_type == 'scale':
            scale = torch.empty(1, device=device).uniform_(0.8, 1.2).item()
            scale = torch.empty(1, device=device).uniform_(0.8, 1.2).item()
            augmented[i] = sample * scale
        elif aug_type == 'invert':
            augmented[i] = -sample
        elif aug_type == 'shift':
            shift = random.randint(-feat_dim // 4, feat_dim // 4)
            augmented[i] = torch.roll(sample, shifts=shift, dims=0)
        elif aug_type == 'channel_dropout':
            n_drop = random.randint(1, min(3, feat_dim // 4))
            drop_idx = torch.randperm(feat_dim, device=device)[:n_drop]
            aug_sample = sample.clone()
            aug_sample[drop_idx] = 0.0
            augmented[i] = aug_sample
        elif aug_type == 'band_mask':
            center = random.randint(feat_dim // 4, 3 * feat_dim // 4)
            half_width = random.randint(4, feat_dim // 2)
            left = max(0, center - half_width)
            right = min(feat_dim, center + half_width)
            aug_sample = torch.zeros_like(sample, device=device)
            aug_sample[left:right] = sample[left:right]
            augmented[i] = aug_sample
        elif aug_type == 'impulse':
            n_imp = random.randint(1, 3)
            pos = torch.randperm(feat_dim, device=device)[:n_imp]
            impulse = torch.randn(n_imp, device=device) * 0.1
            aug_sample = sample.clone()
            aug_sample[pos] += impulse
            augmented[i] = aug_sample
        elif aug_type == 'gain_clip':
            gain = torch.empty(1, device=device).uniform_(0.8, 1.3).item()
            clipped = torch.clamp(sample * gain, min=-3.0, max=3.0)
            augmented[i] = clipped

    return augmented

def compress_targets(
    targets: torch.Tensor,
    num_known_classes: int
    ) -> torch.Tensor:
    K = num_known_classes
    known_part = targets[:, :K]
    unknown_part = targets[:, K:]
    is_unknown = unknown_part.sum(dim=1, keepdim=True)
    new_targets = torch.cat([known_part, is_unknown], dim=1)
    return new_targets

def top_k_gating(weights: torch.Tensor, k: int = 3, temperature: float = 1.0) -> torch.Tensor:
    scaled_weights = weights / temperature
    top_k_values, top_k_indices = torch.topk(scaled_weights, k=k, dim=-1, sorted=False)
    masked_weights = torch.zeros_like(weights)
    batch_indices = torch.arange(weights.size(0), device=weights.device).unsqueeze(-1).expand(-1, k)
    masked_weights.scatter_(1, top_k_indices, torch.ones_like(top_k_values))
    result = weights * masked_weights
    result = F.normalize(result, p=1, dim=-1)
    return result


def label_alignment(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    unique_labels = np.unique(y_true)
    unique_preds = np.unique(y_pred)
    n_true = len(unique_labels)
    n_pred = len(unique_preds)
    conf_matrix = np.zeros((n_pred, n_true), dtype=np.int64)
    for i, pred_label in enumerate(unique_preds):
        for j, true_label in enumerate(unique_labels):
            conf_matrix[i, j] = np.sum((y_pred == pred_label) & (y_true == true_label))
    row_ind, col_ind = linear_sum_assignment(-conf_matrix)
    pred_to_true = {}
    for i in range(len(row_ind)):
        pred_to_true[unique_preds[row_ind[i]]] = unique_labels[col_ind[i]]
    y_pred_aligned = np.array([pred_to_true.get(label, -1) for label in y_pred])
    return y_pred_aligned


def calculate_clustering_metrics(y_true: np.ndarray, 
                                  y_pred: np.ndarray,
                                  print_info: bool = True,
                                  prefix: str = "") -> Dict[str, float]:
    y_pred_aligned = label_alignment(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred_aligned)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    
    metrics = {'ACC': acc, 'NMI': nmi, 'ARI': ari}
    
    if print_info:
        print(f"  {prefix}聚类指标:")
        print(f"    ACC: {acc:.4f}")
        print(f"    NMI: {nmi:.4f}")
        print(f"    ARI: {ari:.4f}")
    
    return metrics


def evaluate_clustering(targets: torch.Tensor,
                        compressed_targets: torch.Tensor,
                        known_outputs: torch.Tensor,
                        unknown_outputs: torch.Tensor,
                        num_known_classes: int,
                        prefix: str = "测试") -> Dict[str, float]:
    compressed_targets_idx = compressed_targets.argmax(dim=1).long()
    _, predicted_unknown = unknown_outputs.max(1)
    
    max_vals, _ = torch.max(known_outputs, dim=1)
    is_predicted_unknown = (known_outputs[:, -1] >= max_vals)
    
    is_truly_unknown = (compressed_targets_idx == num_known_classes)
    
    total_samples = targets.size(0)
    predicted_unknown_count = is_predicted_unknown.sum().item()
    truly_unknown_count = is_truly_unknown.sum().item()
    
    valid_mask = is_predicted_unknown & is_truly_unknown
    valid_unknown_count = valid_mask.sum().item()
    
    misclassified_mask = is_predicted_unknown & (~is_truly_unknown)
    misclassified_known_count = misclassified_mask.sum().item()
    
    if valid_unknown_count > 0:
        original_unknown_part = targets[valid_mask, num_known_classes:]
        y_true = original_unknown_part.argmax(dim=1).cpu().numpy()
        y_pred = predicted_unknown[valid_mask].cpu().numpy()
        
        clustering_metrics = calculate_clustering_metrics(
            y_true, y_pred, 
            print_info=True, 
            prefix=prefix
        )
        print(f"  {prefix}聚类统计: 有效样本={valid_unknown_count}, 被错误分类的已知类={misclassified_known_count}")
        print(f"  {prefix}聚类统计: 预测为未知类={predicted_unknown_count}, 实际未知类={truly_unknown_count}, 总样本={total_samples}")
    else:
        clustering_metrics = {'ACC': 0.0, 'NMI': 0.0, 'ARI': 0.0}
        print(f"  {prefix}聚类指标: 没有满足条件的样本（模型预测为未知类且真实标签为未知类）")
        print(f"  {prefix}聚类统计: 有效样本={valid_unknown_count}, 被错误分类的已知类={misclassified_known_count}")
        print(f"  {prefix}聚类统计: 预测为未知类={predicted_unknown_count}, 实际未知类={truly_unknown_count}, 总样本={total_samples}")
    
    return clustering_metrics


def main():
    torch.manual_seed(42)
    origin_train_epochs = 0
    train_epochs = 50
    train_epochs = 50

    num_known_classes = 6
    num_sum_classes = 10
    num_sum_classes = 10
    num_unknown_classes = num_sum_classes - num_known_classes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(0.95)
        torch.cuda.set_per_process_memory_fraction(0.95)
    
    print("加载数据集...")
    train_dataset = TAUDataset(split='train')
    calib_dataset = TAUDataset(split='calib')
    test_dataset = TAUDataset(split='test')
    
    batch_size = 4096
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        prefetch_factor=8,
        persistent_workers=True,
        drop_last=False
        num_workers=16,
        pin_memory=True,
        prefetch_factor=8,
        persistent_workers=True,
        drop_last=False
    )
    
    calib_loader = DataLoader(
        calib_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        prefetch_factor=8,
        persistent_workers=True,
        drop_last=False
        drop_last=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
        prefetch_factor=8,
        persistent_workers=True,
        drop_last=False
        persistent_workers=True,
        drop_last=False
    )
    
    print(f"Batch size: {batch_size}")
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"校准集样本数: {len(calib_dataset)}")
    print(f"验证集样本数: {len(test_dataset)}")
    
    print("\n创建模型...")
    model = FinalModel(num_unknown_classes=num_unknown_classes, num_known_classes=num_known_classes, 
                        reload_feature_model_pretrained=True)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\n从第",origin_train_epochs,"轮创建训练器...")
    print("\n开始训练...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        calib_loader=calib_loader,
        test_loader=test_loader,
        train_dataset=train_dataset,
        calib_dataset=calib_dataset,
        test_dataset=test_dataset
    )
    
    os.makedirs('experiment/yamnet_C2MoE_2stageN', exist_ok=True)

    trainer.train(epochs=train_epochs)
    
    trainer.plot_history(from_epochs=origin_train_epochs,
                            to_epochs=origin_train_epochs+train_epochs)
    
    best_model_path = 'experiment/yamnet_C2MoE_2stageN/yamnet_C2MoE_2stageN_best.pth'
    final_model_path = 'experiment/yamnet_C2MoE_2stageN/'+\
                'yamnet_C2MoE_2stageN_trained_'+str(origin_train_epochs+train_epochs)+'.pth'
    
    if os.path.exists(best_model_path):
        if os.path.exists(final_model_path):
            os.remove(final_model_path)
        os.rename(best_model_path, final_model_path)
        print(f'\n最佳模型已重命名为: {final_model_path}')
    else:
        model.save(final_model_path)
        print(f'\n模型已保存到: {final_model_path}')
    
    print("\n测试模型加载...")
    loaded_model = FinalModel.load(path = 'experiment/yamnet_C2MoE_2stageN/'+\
                'yamnet_C2MoE_2stageN_trained_'+str(origin_train_epochs+train_epochs)+'.pth')
    print("模型加载成功!")
    
    return trainer, loaded_model

def main_two_stage():
    torch.manual_seed(42)
    
    from_epochs = 0
    stage1_epochs = 60
    stage2_epochs = 60

    num_known_classes = 6
    num_sum_classes = 10
    num_unknown_classes = num_sum_classes - num_known_classes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(0.95)
    
    print("加载数据集...")
    train_dataset = TAUDataset(split='train')
    calib_dataset = TAUDataset(split='calib')
    test_dataset = TAUDataset(split='test')
    
    batch_size = 3000
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        prefetch_factor=8,
        persistent_workers=True,
        drop_last=True
    )
    
    calib_loader = DataLoader(
        calib_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        prefetch_factor=8,
        persistent_workers=True,
        drop_last=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
        prefetch_factor=8,
        persistent_workers=True
    )
    
    print(f"Batch size: {batch_size}")
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"校准集样本数: {len(calib_dataset)}")
    print(f"验证集样本数: {len(test_dataset)}")
    
    print("\n创建模型...")
    model = FinalModel(num_unknown_classes=num_unknown_classes, 
                       num_known_classes=num_known_classes, 
                       reload_feature_model_pretrained=True)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\n创建训练器...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        calib_loader=calib_loader,
        test_loader=test_loader,
        train_dataset=train_dataset,
        calib_dataset=calib_dataset,
        test_dataset=test_dataset
    )
    
    os.makedirs('experiment/yamnet_C2MoE_2stageN', exist_ok=True)

    print("\n" + "=" * 70)
    print("开始两阶段训练")
    print(f"  第一阶段: {stage1_epochs} epochs (分类器训练)")
    print(f"  第二阶段: {stage2_epochs} epochs (聚类器训练)")
    print("=" * 70)
    
    trainer.train_two_stage(stage1_epochs=stage1_epochs, stage2_epochs=stage2_epochs)
    
    total_epochs = stage1_epochs + stage2_epochs
    trainer.plot_history(from_epochs=from_epochs, to_epochs=total_epochs)
    
    stage2_best_path = 'experiment/yamnet_C2MoE_2stageN/stage2_best.pth'
    final_model_path = f'experiment/yamnet_C2MoE_2stageN/yamnet_C2MoE_2stageN_trained_from{str(from_epochs)}to{str(total_epochs)}.pth'
    
    if os.path.exists(stage2_best_path):
        if os.path.exists(final_model_path):
            os.remove(final_model_path)
        os.rename(stage2_best_path, final_model_path)
        print(f'\n最佳模型已保存到: {final_model_path}')
    else:
        print(f'\n警告: 未找到第二阶段最佳模型 {stage2_best_path}')
    
    if os.path.exists(final_model_path):
        print("\n测试模型加载...")
        loaded_model = FinalModel.load(path=final_model_path)
        print("模型加载成功!")
        
        print("\n" + "=" * 70)
        print("最终模型评估")
        print("=" * 70)
        test_loss, test_acc, test_cluster_acc, test_clustering_metrics = trainer.test()
        print(f'\n最终测试结果:')
        print(f'  Test Loss: {test_loss:.4f}')
        print(f'  Test Acc (分类): {test_acc:.2f}%')
        print(f'  Test Cluster Acc (聚类): {test_cluster_acc:.2f}%')
        print(f'  Test NMI: {test_clustering_metrics["NMI"]:.4f}')
        print(f'  Test ARI: {test_clustering_metrics["ARI"]:.4f}')
        
        return trainer, loaded_model
    else:
        print("\n警告: 无法加载最终模型")
        return trainer, model


if __name__ == "__main__":
    print("\n选择训练模式:")
    print("  1 - 单阶段训练 (原始LossF)")
    print("  2 - 两阶段训练 (Stage1Loss + Stage2Loss)")
    
    training_mode = input("\n请输入训练模式 (1 或 2, 默认为2): ").strip()
    
    if training_mode == "1":
        print("\n开始单阶段训练...")
        trainer, model = main()
    else:
        print("\n开始两阶段训练...")
        trainer, model = main_two_stage()
    
    print(f"\n总参数量: {sum(p.numel() for p in model.parameters()):,}")

    print("\n选择训练模式:")
    print("  1 - 单阶段训练 (原始LossF)")
    print("  2 - 两阶段训练 (Stage1Loss + Stage2Loss)")
    
    training_mode = input("\n请输入训练模式 (1 或 2, 默认为2): ").strip()
    
    if training_mode == "1":
        print("\n开始单阶段训练...")
        trainer, model = main()
    else:
        print("\n开始两阶段训练...")
        trainer, model = main_two_stage()
    
    print(f"\n总参数量: {sum(p.numel() for p in model.parameters()):,}")
