class FinalModel(ModelInterface):
    
    def __init__(self, 
                 num_unknown_classes: int = 8,
                 num_known_classes: int = 6,
                 reload_feature_model_pretrained: bool = True):
        super().__init__(num_known_classes)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cpudevice = "cpu"
        self.feature_model = torch_yamnet(pretrained=False)
        self.num_unknown_classes = num_unknown_classes
        self.num_known_classes = num_known_classes

        feature_num = 32
        condition_dim = 0
        self.condition_vector = torch.zeros(condition_dim).to(self.device)

        # 模型参数
        # 加载yamnet预训练模型权重
        if reload_feature_model_pretrained:
            path = 'yamnet.pth'
            state = torch.load(path)
            self.feature_model.load_state_dict(state)

        classifier_cluster_num = 8

        self.fc1 = nn.Sequential(
            nn.Linear(521, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5)
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

        # 门控网络
        self.num_experts = 3  # 减少专家数量，降低复杂度
        self.num_experts_cluster = 3  # 减少专家数量，降低复杂度
        self.classifier_experts = nn.ModuleList([FlowBasedTissue(
                                                input_dim=feature_num,
                                                prototype_num=num_known_classes+1, # 加1表示合并未知类
                                                condition_dim=condition_dim,
                                                num_coupling_layers=3,  # 减少层数，降低复杂度
                                                hidden_dims=[32, 32],  # 减少隐藏层维度
                                                use_permutation=True,
                                                permutation_type='fixed',
                                                cluster_num=classifier_cluster_num # 分类器聚类数
                                                ) for _ in range(self.num_experts)])
        self.gate1 = PolicyNet(feature_num, self.num_experts)

        # 可训练阈值（标量参数）
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = yamnet_infer.waveform_to_log_mel_patches(x, sample_rate=16000)
        
        feature = self.feature_model(x, to_prob=False)
        feature = self.fc1(feature)
        feature = self.fc2(feature)
        feature = self.fc3(feature) # [batch, feature_num]
        batch_size = feature.size(0)
        # print("feature:", feature.shape, feature.dtype)

        # 检查 feature 张量中的异常值
        # try:
        #     check_for_anomalies(feature, "Problematic Tensor")
        # except ValueError as e:
        #     print(f"Caught error: {e}")

        # preds = self.classifier(feature, self.condition_vector)  # [batch, num_known_classes]
        
        weights1 = self.gate1(feature, 1) # [batch, num_experts]
        weights1 = top_k_gating(weights1, k=self.num_experts) # [batch, num_experts] top-k稀疏化
        
        preds = torch.zeros(batch_size, self.num_known_classes+1, device=self.device) # [batch, num_known_classes+1]
        for i in range(self.num_experts):
            expert_output = self.classifier_experts[i](feature, self.condition_vector)
            preds += weights1[:, i].unsqueeze(1) * expert_output # [batch, num_known_classes+1]

        preds = F.softmax(preds, dim=1)  # [batch, num_known_classes+1]

        # preds = known2unknown_probs(preds)  # [batch, known_classes + 1]
        # 好像会崩溃出现NaN，暂时先不加这个了，后续再调试
        # print("preds:", preds.shape, preds.dtype)

        return preds