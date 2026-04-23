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
import fnmatch

import json
import hashlib
import random

class DESEDDataset:
    def __init__(
        self,
        split,
        device = "cpu",
        max_duration = 2,
    ):
        self.device = device
        self.split = split
        self.train_ratio = 0.7
        # self.original_sample_rate = 48000
        # self.target_sample_rate = 16000
        # self.resampler = torchaudio.transforms.Resample(
        #     orig_freq=self.original_sample_rate,
        #     new_freq=self.target_sample_rate
        # ).to(self.device)
        self.sample_rate = 16000
        
        if not os.path.exists('/data/coding/audio_classifying/unprocessed_data'):
            os.makedirs('/data/coding/audio_classifying/unprocessed_data')

        self.dataset_dir = '/data/coding/DESED_synthetic/audio'
        if split in {"train","calib","train_eval"}:
            dataset_path = self.dataset_dir + "/train.csv"
            self.split_dataset_dir = self.dataset_dir + "/train/soundbank/foreground"
            self.save_file_path = '/data/coding/audio_classifying/unprocessed_data/DESED_train_pro_' + f'{max_duration}' + 's.pt'
        elif split == "test":
            dataset_path = self.dataset_dir + "/eval.csv"
            self.split_dataset_dir = self.dataset_dir + "/eval/soundbank/foreground"
            self.save_file_path = '/data/coding/audio_classifying/unprocessed_data/DESED_eval_pro_' + f'{max_duration}' + 's.pt'

        self.max_duration = max_duration
        # self.data_type = ".wav"
        self.saving_audio_batch = 64
        # 读取 label_mapping 字典
        label_mapping_csv = self.dataset_dir + '/vocabulary.csv'
        self.label_mapping = self._build_label_mapping(label_mapping_csv)
        self.num_classes = len(self.label_mapping)

        self.file_to_csv(self.split_dataset_dir,dataset_path)
        self.dataset_map = self.csv_to_dataset(dataset_path)
        # 进度检测
        self.dataset_map_len = len(self.dataset_map)

        if os.path.isfile(self.save_file_path):
            loaded_data = torch.load(self.save_file_path, weights_only=True)
            self.dataset = self._filter_by_split(loaded_data)
        else:
            examples = []
            # padding_mask = torch.zeros(1, 10000).bool()

            for index, (uniq_id, audio_label) in enumerate(tqdm(self.dataset_map)):
                wav_path = os.path.join(self.split_dataset_dir, uniq_id)
                audio_data, _ = sf.read(wav_path, dtype="float32")
                segments = self.audio_postprocess(torch.tensor(audio_data), self.sample_rate, self.max_duration)
                
                label_item = torch.zeros(self.num_classes, dtype=torch.float)
                if audio_label in self.label_mapping:
                    label_id = self.label_mapping[audio_label]
                    label_item[label_id] = 1
                label_item = label_item.unsqueeze(0)

                # processed_segments = self.MFCCprocess(segments, device=self.device)
                
                for processed_segment in segments:
                    example = {
                        "source_audio": processed_segment,
                        "target": label_item,
                        # "padding_mask": padding_mask
                    }
                    examples.append(example)


                if (index + 1) % self.saving_audio_batch == 0 or (index + 1) == len(self.dataset_map):
                    try:
                        # 尝试加载已保存的文件
                        existing_data = torch.load(self.save_file_path, weights_only=True)
                    except FileNotFoundError:
                        # 如果文件不存在，创建一个空列表
                        existing_data = []

                    existing_data.extend(examples)

                    # 保存合并后的数据
                    torch.save(existing_data, self.save_file_path)

                    # 清空当前批次的数据
                    examples = []
                    existing_data = []

                # if index >= 16:
                #     break
                # print(f"index:{index}")
            print(f"处理后的数据已保存到 {self.save_file_path}")
            self.dataset = torch.load(self.save_file_path, weights_only=True)
            print(f"total_num:{len(self.dataset)}")

        # 保存和读取pair_dict
        self.pair_dict_path = self.save_file_path.replace('.pt', '_pairdict.json')
        if split in {"calib", "test"}:
            if os.path.exists(self.pair_dict_path):
                with open(self.pair_dict_path, "r", encoding="utf-8") as f:
                    self.pair_dict = json.load(f)
                # json读取后key是str，转为int
                self.pair_dict = {int(k): v for k, v in self.pair_dict.items()}
            else:
                self.pair_dict = self._build_pair_dict(len(self.dataset), n_pairs=5)
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

    def file_to_csv(self,dataset_dir,dataset_path):
        def match_folder_prefix(folder_name, patterns):
            for pattern in patterns:
                # 支持多种通配符模式
                if fnmatch.fnmatch(folder_name, pattern):
                    # 提取实际匹配部分（保留数字等后缀）
                    return pattern.split('*')[0] if '*' in pattern else pattern
            return None

        # 收集所有文件路径和标签
        records = []
        for subdir in os.listdir(dataset_dir):
            subdir_path = os.path.join(dataset_dir, subdir)
            
            if not os.path.isdir(subdir_path):
                continue  # 跳过非文件夹项
                
            # 提取标签前缀并验证
            label = match_folder_prefix(subdir, self.label_mapping)
            if label is None:  # 无匹配则跳过
                continue
                
            # 遍历子文件夹内所有文件
            for root, _, files in os.walk(subdir_path):
                for file in files:
                    if file.startswith('._') or file.startswith('~$'):  # 排除系统隐藏文件
                        continue
                    # 构建相对路径（统一使用Linux风格路径分隔符）
                    rel_path = os.path.relpath(os.path.join(root, file), dataset_dir)
                    rel_path = rel_path.replace(os.sep, '/')  # 统一路径分隔符
                    
                    records.append({
                        "file_path": rel_path,
                        "label": label
                    })

        # 写入CSV文件
        with open(dataset_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ["file_path", "label"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            writer.writerows(records)

    def csv_to_dataset(self, csv_file_path):
        dataset = []
        with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            # 跳过标题行（如果有）
            next(reader, None)
            for row in reader:
                dataset.append((row[0], row[1]))
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

    def audio_postprocess(self, feats, sample_rate, max_duration, k=0.2):
        """
        处理音频分段：
        1. 短音频：添加均值在k的随机空白段后堆叠分割
        2. 长音频：直接按max_duration分割
        3. 等长音频：直接返回
        
        参数:
            k: 空白段占原音频长度的平均比例
        """
        max_length = int(sample_rate * max_duration)
        segments = []
        original_length = len(feats)
        
        # 短音频处理：添加随机空白段后堆叠
        if original_length < max_length:
            # 计算空白段长度的随机范围（平均值为k）
            min_ratio = max(0.1, k - 0.1)  # 最小比例（不低于0.1）
            max_ratio = min(2.0, k + 0.1)  # 最大比例（不高于2.0）
            
            # 创建空白段+音频的组合段
            combined_segments = []
            while len(combined_segments) < 3:  # 至少生成3个组合段保证多样性
                # 生成随机比例因子（均匀分布）
                random_ratio = torch.empty(1).uniform_(min_ratio, max_ratio).item()
                # 计算随机空白长度
                silence_length = int(original_length * random_ratio)
                silence = torch.zeros(silence_length)
                
                # 构建"空白+音频"组合段
                combined = torch.cat([silence, feats])
                combined_segments.append(combined)
            
            # 堆叠组合段
            stacked = torch.cat(combined_segments)
            
            # 分割堆叠后的音频
            for i in range(0, len(stacked), max_length):
                segment = stacked[i:i + max_length]
                if len(segment) < max_length:
                    padding = torch.zeros(max_length - len(segment))
                    segment = torch.cat([segment, padding])
                segments.append(segment)
        
        # 长音频处理：直接分割
        elif original_length > max_length:
            for i in range(0, original_length, max_length):
                segment = feats[i:i + max_length]
                if len(segment) == max_length:
                    segments.append(segment)
        
        # 等长音频处理
        else:
            segments.append(feats)
            
        return segments

    def _filter_by_split(self, data):
        if self.split == "test":
            return data  # 测试集直接返回全部
        
        # 对train/calib数据重新划分（保持与原始划分一致）
        from sklearn.model_selection import train_test_split
        indices = list(range(len(data)))
        train_idx, calib_idx = train_test_split(
            indices,
            train_size=self.train_ratio,
            random_state=42
        )
        
        if self.split == "train":
            return [data[i] for i in train_idx]
        elif self.split == "calib":
            return [data[i] for i in calib_idx]
        elif self.split == "train_eval":
            return data

    def __getitem__(self, index):
        item = self.dataset[index]
        audio = item["source_audio"].clone()
        label = item["target"].clone()

        augtype = np.random.randint(0, 9)  # 0~8 共9种增强方式

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

        item["source_audio"] = augmented_audio
        item["idx"] = index
        return item

    def __len__(self):
        return len(self.dataset)

    def MFCCprocess(
            self,
            source: list,
            device,
            fbank_mean: float = 13.82936,
            fbank_std: float = 8.30694,
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
            # waveform = self.resampler(waveform)
            fbank = ta_kaldi.fbank(waveform, num_mel_bins=num_mel_bins, sample_frequency=sample_frequency, frame_length=frame_length, frame_shift=frame_shift)
            # fbank = ta_kaldi.fbank(waveform, num_mel_bins=128, sample_frequency=16000, frame_length=50, frame_shift=25)
            
            ######## TAU 的采样率为48000 进行重采样########
            
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

    # def preprocess(
    #         self,
    #         source: list,
    #         device,
    #         fbank_mean: float = 9.44335,
    #         fbank_std: float = 5.93001,
    #         num_components: int = 64  # 默认保留64个主成分
    # ) -> torch.Tensor:
    #     fbanks = []
    #     # 将波形数据从浮点数转换为整数表示，将波形数据转换为 MFCC 特征并添加到列表中
    #     for waveform in source:
    #         waveform = torch.tensor(waveform).to(device).unsqueeze(0) * 2 ** 15
    #         # 这里假设 ta_kaldi 是已经正确导入的模块
    #         # 将原始音频波形的采样率转换为 16,000Hz ，并使用 25ms 的 Povey 窗口（每 10ms 移动一次）提取 128 维的 Mel 滤波器组特征，作为声学特征。
    #         # waveform = self.resampler(waveform)
    #         fbank = ta_kaldi.fbank(waveform, num_mel_bins=128, sample_frequency=self.sample_rate, frame_length=25, frame_shift=10)
    #         # fbank = ta_kaldi.fbank(waveform, num_mel_bins=128, sample_frequency=16000, frame_length=50, frame_shift=25)
            
    #         ######## TAU 的采样率为48000 ########
            
    #         fbanks.append(fbank)
    #     # MFCC 特征堆叠成一个张量
    #     fbank = torch.stack(fbanks, dim=0)
    #     # 归一化处理
    #     fbank = (fbank - fbank_mean) / (2 * fbank_std)  # fbank=[batch, num_subpovey, num_mel_bins=128]

    #     # 进行主成分分析
    #     batch_size, num_frames, num_features = fbank.shape
    #     fbank_reshaped = fbank.reshape(-1, num_features)
    #     # 计算协方差矩阵
    #     mean = torch.mean(fbank_reshaped, dim=0)
    #     centered = fbank_reshaped - mean
    #     cov_matrix = torch.matmul(centered.T, centered) / (centered.shape[0] - 1)

    #     # 计算特征值和特征向量
    #     eigenvalues, eigenvectors = LA.eigh(cov_matrix)

    #     # 按特征值从大到小排序
    #     sorted_indices = torch.argsort(eigenvalues, descending=True)
    #     sorted_eigenvectors = eigenvectors[:, sorted_indices]

    #     # 选择前 num_components 个主成分
    #     top_eigenvectors = sorted_eigenvectors[:, :num_components]

    #     # 投影到主成分上
    #     fbank_pca = torch.matmul(centered, top_eigenvectors)

    #     # 恢复形状
    #     fbank_pca = fbank_pca.reshape(batch_size, num_frames, num_components).to('cpu')

    #     return fbank_pca

# 使用示例
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = DESEDDataset(split="train_eval", device=device)
    test_dataset = DESEDDataset(split="test", device=device)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    for batch_idx, batch in enumerate(tqdm(train_loader)):
        audio, label = batch["source_audio"].to(device), batch["target"].to(device)
        print(f"audio:{audio.shape}") ## [64, 201, 128]
        break

    all_fbanks = []
    for item in train_dataset.dataset:
        # 假设 item["source_audio"] 是 MFCC 特征
        fbank = item["source_audio"]
        all_fbanks.append(fbank)

    all_fbanks = torch.cat(all_fbanks, dim=0)  # [total_frames, num_mel_bins]
    fbank_mean = all_fbanks.mean().item()
    fbank_std = all_fbanks.std().item()
    print(f"train_dataset_fbank_mean: {fbank_mean}, fbank_std: {fbank_std}")

    all_fbanks = []
    for item in test_dataset.dataset:
        # 假设 item["source_audio"] 是 MFCC 特征
        fbank = item["source_audio"]
        all_fbanks.append(fbank)

    all_fbanks = torch.cat(all_fbanks, dim=0)  # [total_frames, num_mel_bins]
    fbank_mean = all_fbanks.mean().item()
    fbank_std = all_fbanks.std().item()
    print(f"test_dataset_fbank_mean: {fbank_mean}, fbank_std: {fbank_std}")
    