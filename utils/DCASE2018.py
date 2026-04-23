import torch
import numpy as np
import os
import pandas as pd
from tqdm import tqdm  
import csv
import torch.linalg as LA
import soundfile as sf
import torchaudio.compliance.kaldi as ta_kaldi
import torch.utils.data as data
import torchaudio

import json
import hashlib
import random

class D2018Dataset:
    def __init__(
        self,
        split,
        device = "cpu",
        fold = 2,
        max_duration = 2, ###2可运行
    ):
        self.device = device
        self.split = split
        self.train_ratio = 0.7
        self.original_sample_rate = 48000
        self.target_sample_rate = 16000
        self.resampler = torchaudio.transforms.Resample(
            orig_freq=self.original_sample_rate,
            new_freq=self.target_sample_rate
        ).to(self.device)

        if not os.path.exists('/data/coding/audio_classifying/unprocessed_data'):
            os.makedirs('/data/coding/audio_classifying/unprocessed_data')

        if split in {"train", "train_eval"}:
            dataset_path = "/data/coding/DCASE2018-task5-dev/evaluation_setup/fold"+ f"{fold}" +"_train.txt"
            self.save_file_path = '/data/coding/audio_classifying/unprocessed_data/D2018_train_pro_fold' + f'{fold}' + '_' + f'{max_duration}' + 's.pt'
        elif split in {"calib", "test"}:
            dataset_path = "/data/coding/DCASE2018-task5-dev/evaluation_setup/fold"+ f"{fold}" +"_evaluate.txt"
            self.save_file_path = '/data/coding/audio_classifying/unprocessed_data/D2018_eval_pro_fold' + f'{fold}' + '_' + f'{max_duration}' + 's.pt'

        self.metafile = "/data/coding/DCASE2018-task5-dev/meta.txt"
        self.dataset_map = self.txt_to_dataset(dataset_path, self.metafile)
        # 进度检测
        self.dataset_map_len = len(self.dataset_map)
        self.max_duration = max_duration
        self.dataset_dir = "/data/coding/DCASE2018-task5-dev"
        # 读取 label_mapping 字典
        label_mapping_csv = '/data/coding/DCASE2018-task5-dev/evaluation_setup/vocabulary.csv'
        self.label_mapping = self._build_label_mapping(label_mapping_csv)
        self.num_classes = len(self.label_mapping)

        # 保存和读取pair_dict
        self.pair_dict_path = self.save_file_path.replace('.pt', '_pairdict.json')
        if split in {"calib", "test"}:
            if os.path.exists(self.pair_dict_path):
                with open(self.pair_dict_path, "r", encoding="utf-8") as f:
                    self.pair_dict = json.load(f)
                # json读取后key是str，转为int
                self.pair_dict = {int(k): v for k, v in self.pair_dict.items()}
            else:
                self.pair_dict = self._build_pair_dict(len(self.dataset_map), n_pairs=5)
                # json要求key为str
                with open(self.pair_dict_path, "w", encoding="utf-8") as f:
                    json.dump({str(k): v for k, v in self.pair_dict.items()}, f, ensure_ascii=False, indent=2)
        else:
            self.pair_dict = None

    def _build_pair_dict(self, total, n_pairs=5):
        pair_dict = {}
        for idx_anchor in tqdm(range(total)):
            candidate_idx = [i for i in range(total) if i != idx_anchor]
            seed = int(hashlib.md5(str(idx_anchor).encode()).hexdigest(), 16) % (2**32)
            rng = random.Random(seed)
            sampled_idx = rng.sample(candidate_idx, n_pairs)
            pair_dict[idx_anchor] = sampled_idx
        return pair_dict

    def txt_to_dataset(self, txt_file_path, meta_path):
        dataset = []
        # 如果是 test split，需要从 meta.txt 获取标签
        meta_dict = {}
        with open(txt_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    dataset.append((parts[0], parts[1]))
        return dataset

    def _build_label_mapping(self, csv_file):
        label_mapping = {}
        try:
            df = pd.read_csv(csv_file, header=None)
            for _, row in df.iterrows():
                label_mapping[row.iloc[1]] = row.iloc[0]
        except FileNotFoundError:
            print(f"错误：未找到 {csv_file} 文件。")
        return label_mapping

    def audio_postprocess(self, feats, sample_rate, max_duration):
        max_length = int(sample_rate * max_duration)
        if len(feats) > max_length:
            start = np.random.randint(0, len(feats) - max_length)
            segment = feats[start:start + max_length]
        elif len(feats) < max_length:
            padding = torch.zeros(max_length - len(feats))
            segment = torch.cat([feats, padding])
        else:
            segment = feats
        return segment

    def __getitem__(self, index):
        uniq_id, audio_label = self.dataset_map[index]
        wav_path = os.path.join(self.dataset_dir, uniq_id)
        audio_data, _ = sf.read(wav_path, dtype="float32")
        audio = torch.tensor(audio_data[:, 0])
        audio = self.audio_postprocess(audio, self.original_sample_rate, self.max_duration)

        augtype = np.random.randint(0, 9)
        max_length = audio.shape[0]

        if augtype == 0:
            # 原始
            augmented_audio = audio
        elif augtype == 1:
            # 高斯噪声
            noise = torch.randn_like(audio) * 0.01
            augmented_audio = audio + noise
        elif augtype == 2:
            # 均匀噪声
            uniform_noise = (torch.rand_like(audio) - 0.5) * 0.02
            augmented_audio = audio + uniform_noise
        elif augtype == 3:
            # 简单混响
            rir = torch.zeros(32, device=audio.device)
            rir[0] = 1
            rir[10] = 0.5
            reverb = torch.nn.functional.conv1d(audio.unsqueeze(0).unsqueeze(0), rir.unsqueeze(0).unsqueeze(0), padding=16)
            augmented_audio = reverb.squeeze()[:max_length]
        elif augtype == 4:
            # 随机让两段各2%的片段变为空白（静音）
            cut = int(0.02 * max_length)
            augmented_audio = audio.clone()
            for _ in range(2):
                start = np.random.randint(0, max_length - cut)
                augmented_audio[start:start+cut] = 0
        elif augtype == 5:
            # 随机增益
            gain = np.random.uniform(0.7, 1.3)
            augmented_audio = audio * gain
        elif augtype == 6:
            # 高斯+混响
            noise = torch.randn_like(audio) * 0.01
            rir = torch.zeros(32, device=audio.device)
            rir[0] = 1
            rir[10] = 0.5
            reverb = torch.nn.functional.conv1d((audio + noise).unsqueeze(0).unsqueeze(0), rir.unsqueeze(0).unsqueeze(0), padding=16)
            augmented_audio = reverb.squeeze()[:max_length]
        elif augtype == 7:
            # 均匀噪声+增益
            uniform_noise = (torch.rand_like(audio) - 0.5) * 0.02
            gain = np.random.uniform(0.7, 1.3)
            augmented_audio = (audio + uniform_noise) * gain
        elif augtype == 8:
            # 高斯+均匀噪声+混响
            noise = torch.randn_like(audio) * 0.01
            uniform_noise = (torch.rand_like(audio) - 0.5) * 0.02
            rir = torch.zeros(32, device=audio.device)
            rir[0] = 1
            rir[10] = 0.5
            noisy_audio = audio + noise + uniform_noise
            reverb = torch.nn.functional.conv1d(noisy_audio.unsqueeze(0).unsqueeze(0), rir.unsqueeze(0).unsqueeze(0), padding=16)
            augmented_audio = reverb.squeeze()[:max_length]

        label_item = torch.zeros(self.num_classes, dtype=torch.float)
        if audio_label in self.label_mapping:
            label_id = self.label_mapping[audio_label]
            label_item[label_id] = 1
        label_item = label_item.unsqueeze(0)

        item = {
            "source_audio": augmented_audio,
            "target": label_item,
            "idx": index
        }
        return item

    def __len__(self):
        return len(self.dataset_map)

    def MFCCprocess(
            self,
            source: list,
            device,
            fbank_mean: float = 11.19348,
            fbank_std: float = 3.41163,
            num_mel_bins: int = 128,
            sample_frequency: int = 16000,
            frame_length: int = 25,
            frame_shift: int = 10
        ) -> torch.Tensor:
        fbanks = []
        # 将波形数据从浮点数转换为整数表示，将波形数据转换为 MFCC 特征并添加到列表中
        for waveform in source:
            waveform = torch.tensor(waveform).to(device).unsqueeze(0) * 2 ** 15
            # 这里假设 ta_kaldi 是已经正确导入的模块
            # 将原始音频波形的采样率转换为 16,000Hz ，并使用 25ms 的 Povey 窗口（每 10ms 移动一次）提取 128 维的 Mel 滤波器组特征，作为声学特征。
            waveform = self.resampler(waveform)
            fbank = ta_kaldi.fbank(waveform, num_mel_bins=num_mel_bins, sample_frequency=sample_frequency, frame_length=frame_length, frame_shift=frame_shift)
            # fbank = ta_kaldi.fbank(waveform, num_mel_bins=128, sample_frequency=16000, frame_length=50, frame_shift=25)
            
            fbanks.append(fbank)
        # MFCC 特征堆叠成一个张量
        fbank = torch.stack(fbanks, dim=0)
        # 归一化处理
        fbank = (fbank - fbank_mean) / (2 * fbank_std)
        return fbank.to('cpu')  # fbank=[batch, frames, num_mel_bins=128]

        # # 堆叠为 batch
        # waveforms = [torch.tensor(w).to(device).unsqueeze(0) * 2 ** 15 for w in source]
        # waveforms = torch.cat(waveforms, dim=0)  # [batch, 1, samples]
        # # waveforms = self.resampler(waveforms)    # [batch, 1, samples] or [batch, samples]
        # # torchaudio.transforms.MFCC 支持批量和GPU
        # mfcc_transform = torchaudio.transforms.MFCC(
        #     sample_rate=sample_frequency,
        #     n_mfcc=num_mel_bins,
        #     melkwargs={
        #         'n_fft': 2048,
        #         'hop_length': int(sample_frequency * frame_shift / 1000),
        #         'win_length': int(sample_frequency * frame_length / 1000),
        #         'center': True,
        #         'power': 2.0,
        #     }
        # ).to(device)
        # mfcc = mfcc_transform(waveforms)  # [batch, n_mfcc, frames]
        # # mfcc = (mfcc - fbank_mean) / fbank_std
        # return mfcc.transpose(1, 2).cpu()  # fbank=[batch, frames, num_mel_bins=128]

    def preprocess(
            self,
            source: list,
            device,
            fbank_mean: float = 0.24481,
            fbank_std: float = 0.58659,
            num_components: int = 64  # 默认保留64个主成分
    ) -> torch.Tensor:
        fbanks = []
        # 将波形数据从浮点数转换为整数表示，将波形数据转换为 MFCC 特征并添加到列表中
        for waveform in source:
            waveform = torch.tensor(waveform).to(device).unsqueeze(0) * 2 ** 15
            # 这里假设 ta_kaldi 是已经正确导入的模块
            # 将原始音频波形的采样率转换为 16,000Hz ，并使用 25ms 的 Povey 窗口（每 10ms 移动一次）提取 128 维的 Mel 滤波器组特征，作为声学特征。
            waveform = self.resampler(waveform)
            fbank = ta_kaldi.fbank(waveform, num_mel_bins=128, sample_frequency=16000, frame_length=25, frame_shift=10)
            # fbank = ta_kaldi.fbank(waveform, num_mel_bins=128, sample_frequency=16000, frame_length=50, frame_shift=25)
            
            fbanks.append(fbank)
        # MFCC 特征堆叠成一个张量
        fbank = torch.stack(fbanks, dim=0)
        # 归一化处理
        fbank = (fbank - fbank_mean) / (2 * fbank_std)  # fbank=[batch, num_subpovey, num_mel_bins=128]

        # 进行主成分分析
        batch_size, num_frames, num_features = fbank.shape
        fbank_reshaped = fbank.reshape(-1, num_features)
        # 计算协方差矩阵
        mean = torch.mean(fbank_reshaped, dim=0)
        centered = fbank_reshaped - mean
        cov_matrix = torch.matmul(centered.T, centered) / (centered.shape[0] - 1)

        # 计算特征值和特征向量
        eigenvalues, eigenvectors = LA.eigh(cov_matrix)

        # 按特征值从大到小排序
        sorted_indices = torch.argsort(eigenvalues, descending=True)
        sorted_eigenvectors = eigenvectors[:, sorted_indices]

        # 选择前 num_components 个主成分
        top_eigenvectors = sorted_eigenvectors[:, :num_components]

        # 投影到主成分上
        fbank_pca = torch.matmul(centered, top_eigenvectors)

        # 恢复形状
        fbank_pca = fbank_pca.reshape(batch_size, num_frames, num_components).to('cpu')

        return fbank_pca

# 使用示例
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = D2018Dataset(split="train_eval", device=device)
    test_dataset = D2018Dataset(split="test", max_duration=10, device=device) 
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    for batch_idx, batch in enumerate(tqdm(train_loader)):
        audio, label = batch["source_audio"].to(device), batch["target"].to(device)
        print(f"audio:{audio.shape}") ## [64, 201, 128]
        break

    