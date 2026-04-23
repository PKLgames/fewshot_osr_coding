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

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


from utils.TAU22 import TAUDataset
import yamnet_PT_inference as yamnet_infer
from torch_audioset.yamnet.model import yamnet as torch_yamnet
from Flow_Models import FlowBasedTissue
from loss_function.loss_function_osr import LossF

from exam_tool import check_for_anomalies
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

class ModelInterface(nn.Module):
    def __init__(self, num_known_classes: int = 10):
        super().__init__()
        self.num_known_classes = num_known_classes
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def get_config(self) -> Dict[str, Any]:
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
        self.bn2 = nn.BatchNorm1d(32)
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
                 num_known_classes: int = 6,
                 reload_feature_model_pretrained: bool = True):
        super().__init__(num_known_classes)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cpudevice = "cpu"
        self.feature_model = torch_yamnet(pretrained=False)
        self.num_known_classes = num_known_classes
        self.reload_feature_model_pretrained = reload_feature_model_pretrained

        feature_num = 32
        condition_dim = 0
        self.condition_vector = torch.zeros(condition_dim).to(self.device)

        if reload_feature_model_pretrained:
            path = 'yamnet.pth'
            state = torch.load(path)
            self.feature_model.load_state_dict(state)

        classifier_cluster_num = 8

        self.fc1 = nn.Sequential(
            nn.Linear(521, 128),
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
        self.classifier_experts = nn.ModuleList([FlowBasedTissue(
                                                input_dim=feature_num,
                                                prototype_num=num_known_classes+1,
                                                condition_dim=condition_dim,
                                                num_coupling_layers=3,
                                                hidden_dims=[32, 32],
                                                use_permutation=True,
                                                permutation_type='fixed',
                                                cluster_num=classifier_cluster_num
                                                ) for _ in range(self.num_experts)])
        self.gate = PolicyNet(feature_num, self.num_experts)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = yamnet_infer.waveform_to_log_mel_patches(x, sample_rate=16000)
        
        feature = self.feature_model(x, to_prob=False)
        feature = self.fc1(feature)
        feature = self.fc2(feature)
        feature = self.fc3(feature)
        batch_size = feature.size(0)

        weights = self.gate(feature, 1)
        weights = top_k_gating(weights, k=self.num_experts)
        
        preds = torch.zeros(batch_size, self.num_known_classes+1, device=self.device)
        for i in range(self.num_experts):
            expert_output = self.classifier_experts[i](feature, self.condition_vector)
            preds += weights[:, i].unsqueeze(1) * expert_output

        preds = F.softmax(preds, dim=1)

        return preds, weights
    
    def get_config(self) -> Dict[str, Any]:
        return {
            'num_known_classes': self.num_known_classes,
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
        
        self.accumulation_steps = 1

        self.num_known_classes = self.model.num_known_classes
        
        self.model_criterion = LossF(
            lambda_ce=1.5, lambda_binary=1.0, lambda_unk_enc=0.8,
            lambda_conf_penalty=0.1, lambda_expert=0.1,
            lambda_max_conf=0.5,
            label_smoothing=0.25, focal_gamma=2.0
        )
        
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=0.0001,
            weight_decay=2e-3,
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
            'train_loss': [],
            'calib_loss': [],
            'train_acc': [],
            'calib_acc': [],
            'test_loss': [],
            'test_acc': [],
        }

    def train_epoch(self, epoch: int = 0, total_epochs: int = 30, 
                    optimizer = None) -> Tuple[float, float]:
        if optimizer is None:
            optimizer = self.optimizer

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

            known_outputs, weights1 = self.model(inputs)

            train_loss, _ = self.model_criterion(
                known_outputs, compressed_targets, weights1,
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

    def calib_epoch(self, epoch: int = 0, total_epochs: int = 30, 
                    optimizer = None) -> Tuple[float, float]:
        if optimizer is None:
            optimizer = self.optimizer
            
        if len(self.calib_loader) == 0:
            print(f'  Warning: Calibration loader is empty. Skipping this epoch.')
            return 0.0, 0.0
            
        self.model.train()
        calib_running_loss = 0.0
        calib_correct = 0
        calib_total = 0
        
        optimizer.zero_grad()
        
        for batch_idx, item in enumerate(self.calib_loader):
            inputs, targets = item['source_audio'].to(self.device), item['target'].squeeze(1).to(self.device)
            compressed_targets_onehot = compress_targets(targets, self.num_known_classes)
            compressed_targets_idx = compressed_targets_onehot.argmax(dim=1).long()

            known_outputs, weights1 = self.model(inputs)

            calib_loss, _ = self.model_criterion(
                known_outputs, compressed_targets_idx, weights1,
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

            if batch_idx % 100 == 99:
                print(f'  Batch: {batch_idx+1}, Calib_Loss: {calib_loss.item() * self.accumulation_steps:.4f}')
        
        calib_epoch_loss = calib_running_loss / len(self.calib_loader)
        calib_epoch_acc = 100. * calib_correct / calib_total
        
        return calib_epoch_loss, calib_epoch_acc

    def test(self) -> Tuple[float, float]:
        if len(self.test_loader) == 0:
            print(f'  Warning: Test loader is empty. Skipping evaluation.')
            return 0.0, 0.0
            
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, item in enumerate(self.test_loader):
                inputs = item['source_audio'].to(self.device, non_blocking=True)
                targets_onehot = item['target'].squeeze(1).to(self.device, non_blocking=True)
                
                compressed_targets_onehot = compress_targets(targets_onehot, self.num_known_classes)
                compressed_targets_idx = compressed_targets_onehot.argmax(dim=1).long()

                known_outputs, weights1 = self.model(inputs)

                loss, _ = self.model_criterion(
                    known_outputs, compressed_targets_idx, weights1,
                    num_known_classes=self.num_known_classes
                )
                
                running_loss += loss.item()
                total += compressed_targets_idx.size(0)
                _, predicted = known_outputs.max(1)
                correct += predicted.eq(compressed_targets_idx).sum().item()
        
        test_loss = running_loss / len(self.test_loader)
        test_acc = 100. * correct / total
        
        return test_loss, test_acc
    
    def train(self, epochs: int = 50):
        print(f'开始开放集识别训练，设备: {self.device}')
        print('=' * 70)
        
        
        self.best_val_loss = float('inf')
        self.early_stop_counter = 0
        
        # 更新scheduler的T_max为实际epochs数
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-7
        )
        
        for epoch in range(epochs):
            print(f'\nEpoch {epoch+1}/{epochs}')
            print('-' * 50)
            
            train_loss, train_acc = self.train_epoch(
                epoch=epoch, 
                total_epochs=epochs, 
                optimizer=self.optimizer
            )
            
            calib_loss, calib_acc = self.calib_epoch(
                epoch=epoch, 
                total_epochs=epochs,
                optimizer=self.optimizer
            )
            
            test_loss, test_acc = self.test()
            
            self.scheduler.step()
            
            self.history['train_loss'].append(train_loss)
            self.history['calib_loss'].append(calib_loss)
            self.history['train_acc'].append(train_acc)
            self.history['calib_acc'].append(calib_acc)
            self.history['test_loss'].append(test_loss)
            self.history['test_acc'].append(test_acc)
            
            print(f'训练 Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%')
            print(f'校准 Loss: {calib_loss:.4f}, Acc: {calib_acc:.2f}%')
            print(f'验证 Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%')
            
            gap = train_acc - test_acc
            print(f'  >> 过拟合差距: {gap:.2f}%')
            
            if test_loss < self.best_val_loss:
                self.best_val_loss = test_loss
                self.early_stop_counter = 0
                self.model.save('experiment/yamnet_OSR/osr_best.pth')
                print(f'  保存最佳模型 (Val Loss: {test_loss:.4f})')
            else:
                self.early_stop_counter += 1
                print(f'  早停计数: {self.early_stop_counter}/{self.patience}')
                if self.early_stop_counter >= self.patience:
                    print(f'\n!!! 早停触发，停止训练 !!!')
                    break
        
        print('\n' + '=' * 70)
        print('训练完成！')
        print('=' * 70)

    def plot_history(self, from_epochs: int, to_epochs: int):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        axes[0].plot(self.history['train_loss'], label='Train Loss')
        axes[0].plot(self.history['calib_loss'], label='Calib Loss')
        axes[0].plot(self.history['test_loss'], label='Test Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('OSR Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        axes[1].plot(self.history['train_acc'], label='Train Acc')
        axes[1].plot(self.history['calib_acc'], label='Calib Acc')
        axes[1].plot(self.history['test_acc'], label='Test Acc')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('OSR Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(f'experiment/yamnet_OSR/osr_trained_from{str(from_epochs)}to{str(to_epochs)}.png')
        print(f'\n图表已保存: experiment/yamnet_OSR/osr_trained_from{str(from_epochs)}to{str(to_epochs)}.png')


def compress_targets(
    targets: torch.Tensor,
    num_known_classes: int
    ) -> torch.Tensor:
    """将完整标签压缩为已知类+未知类（OSR格式）"""
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


def main():
    torch.manual_seed(42)
    origin_train_epochs = 0
    train_epochs = 50

    num_known_classes = 6
    num_sum_classes = 10

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
    )
    
    print(f"Batch size: {batch_size}")
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"校准集样本数: {len(calib_dataset)}")
    print(f"验证集样本数: {len(test_dataset)}")
    
    print("\n创建OSR模型...")
    model = FinalModel(num_known_classes=num_known_classes, 
                        reload_feature_model_pretrained=True)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\n从第", origin_train_epochs, "轮创建训练器...")
    print("\n开始开放集识别训练...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        calib_loader=calib_loader,
        test_loader=test_loader,
        train_dataset=train_dataset,
        calib_dataset=calib_dataset,
        test_dataset=test_dataset
    )
    
    os.makedirs('experiment/yamnet_OSR', exist_ok=True)

    trainer.train(epochs=train_epochs)
    
    trainer.plot_history(from_epochs=origin_train_epochs,
                            to_epochs=origin_train_epochs+train_epochs)
    
    best_model_path = 'experiment/yamnet_OSR/osr_best.pth'
    final_model_path = 'experiment/yamnet_OSR/'+\
                'osr_trained_'+str(origin_train_epochs+train_epochs)+'.pth'
    
    if os.path.exists(best_model_path):
        if os.path.exists(final_model_path):
            os.remove(final_model_path)
        os.rename(best_model_path, final_model_path)
        print(f'\n最佳模型已重命名为: {final_model_path}')
    else:
        model.save(final_model_path)
        print(f'\n模型已保存到: {final_model_path}')
    
    print("\n测试模型加载...")
    loaded_model = FinalModel.load(path = 'experiment/yamnet_OSR/'+\
                'osr_trained_'+str(origin_train_epochs+train_epochs)+'.pth')
    print("模型加载成功!")
    
    return trainer, loaded_model


if __name__ == "__main__":
    print("\n开始开放集识别(OSR)训练...")
    trainer, model = main()
    print(f"\n总参数量: {sum(p.numel() for p in model.parameters()):,}")