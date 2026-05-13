import torch
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import csv
import librosa


class DCASE18Dataset:
    def __init__(
        self,
        split,
        fold=1,
        max_duration=1,
        use_cache=False,
    ):
        """
        DCASE 2018 Task 5 dataset class for few-shot open-set recognition.

        Mirrors TAU22's TAUDataset output format:
            {"source_audio": tensor, "target": one-hot tensor, "idx": int}

        Parameters
        ----------
        split : str — "train", "calib", or "test"
        fold : int — which cross-validation fold to use (1-4)
        max_duration : float — seconds per audio segment (original is 10s)
        use_cache : bool — if True, pre-process and cache to .pt files
        """
        self.split = split
        self.fold = fold
        self.use_cache = use_cache
        self.target_sample_rate = 16000
        self.max_duration = max_duration

        base_dir = "/coding/DCASE2018-task5-dev"
        cache_dir = "/coding/audio_classifying/unprocessed_data"
        os.makedirs(cache_dir, exist_ok=True)

        # Map split to CSV file
        if split == "train":
            csv_path = f"{base_dir}/sampled_setup/fold{fold}_train.csv"
            cache_file = f"{cache_dir}/DCASE18_fold{fold}_train_{max_duration}s.pt"
        elif split == "calib":
            csv_path = f"{base_dir}/sampled_setup/fold{fold}_calib.csv"
            cache_file = f"{cache_dir}/DCASE18_fold{fold}_calib_{max_duration}s.pt"
        elif split == "test":
            csv_path = f"{base_dir}/sampled_setup/fold{fold}_evaluate.csv"
            cache_file = f"{cache_dir}/DCASE18_fold{fold}_eval_{max_duration}s.pt"
        else:
            raise ValueError(f"Unknown split: {split}")

        self.audio_dir = f"{base_dir}/audio"
        self.base_dir = base_dir

        # Build label mapping from vocabulary
        vocab_csv = f"{base_dir}/sampled_setup/vocabulary.csv"
        self.label_mapping = self._build_label_mapping(vocab_csv)
        self.num_classes = len(self.label_mapping)
        self.class_names = list(self.label_mapping.keys())

        # Read CSV into records list
        self.records = self._read_csv(csv_path)
        self.saving_audio_batch = 64

        if use_cache:
            if os.path.isfile(cache_file):
                self.dataset = torch.load(cache_file, weights_only=True)
            else:
                examples = []
                for idx, (filename, scene_label) in enumerate(tqdm(self.records, desc=f"Caching {split}")):
                    wav_path = os.path.join(self.base_dir, filename)
                    audio_data, _ = librosa.load(wav_path, dtype="float32", sr=self.target_sample_rate, mono=True)
                    segments = self._segment_audio(torch.tensor(audio_data))

                    label_onehot = torch.zeros(self.num_classes, dtype=torch.float)
                    if scene_label in self.label_mapping:
                        label_onehot[self.label_mapping[scene_label]] = 1
                    label_onehot = label_onehot.unsqueeze(0)

                    for seg in segments:
                        examples.append({"source_audio": seg, "target": label_onehot})

                    if (idx + 1) % self.saving_audio_batch == 0 or (idx + 1) == len(self.records):
                        existing = torch.load(cache_file, weights_only=True) if os.path.isfile(cache_file) else []
                        existing.extend(examples)
                        torch.save(existing, cache_file)
                        examples = []

                print(f"Cached data saved to {cache_file}")
                self.dataset = torch.load(cache_file, weights_only=True)
        else:
            self.dataset = None
            self.index_mapping = self._build_index_mapping()
            print(f"Dynamic-load mode: {len(self.index_mapping)} segments across {len(self.records)} files")

    def _read_csv(self, csv_path):
        records = []
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader, None)  # skip header
            for row in reader:
                records.append((row[0], row[1]))
        return records

    def _build_label_mapping(self, vocab_csv):
        df = pd.read_csv(vocab_csv, header=None)
        return {row.iloc[1]: row.iloc[0] for _, row in df.iterrows()}

    def _segment_audio(self, audio):
        max_len = int(self.target_sample_rate * self.max_duration)
        segments = []
        if len(audio) > max_len:
            for i in range(0, len(audio), max_len):
                seg = audio[i:i + max_len]
                if len(seg) == max_len:
                    segments.append(seg)
        elif len(audio) < max_len:
            pad = torch.zeros(max_len - len(audio))
            segments.append(torch.cat([audio, pad]))
        else:
            segments.append(audio)
        return segments

    def _build_index_mapping(self):
        # DCASE2018 audio files are exactly 10s at 16kHz = 160000 samples.
        # Compute segments without loading audio.
        full_samples = 10 * self.target_sample_rate
        seg_samples = int(self.target_sample_rate * self.max_duration)
        if full_samples > seg_samples:
            num_segs = full_samples // seg_samples
        else:
            num_segs = 1

        mapping = []
        for wav_name, scene_label in self.records:
            wav_path = os.path.join(self.base_dir, wav_name)
            for seg_idx in range(num_segs):
                mapping.append((wav_path, scene_label, seg_idx))
        return mapping

    def __getitem__(self, index):
        if self.use_cache:
            item = self.dataset[index]
            audio = item["source_audio"].clone()
            label = item["target"].clone()
        else:
            wav_path, scene_label, seg_idx = self.index_mapping[index]
            audio_data, _ = librosa.load(wav_path, dtype="float32", sr=self.target_sample_rate, mono=True)
            audio_tensor = torch.tensor(audio_data)
            segments = self._segment_audio(audio_tensor)
            audio = segments[seg_idx]

            label = torch.zeros(self.num_classes, dtype=torch.float)
            if scene_label in self.label_mapping:
                label[self.label_mapping[scene_label]] = 1
            label = label.unsqueeze(0)

        return {
            "source_audio": audio,
            "target": label,
            "idx": index,
        }

    def __len__(self):
        if self.use_cache:
            return len(self.dataset)
        return len(self.index_mapping)

    def get_class_names(self):
        return list(self.label_mapping.keys())


if __name__ == "__main__":
    for fold in [1, 2, 3, 4]:
        print(f"\n{'=' * 50}")
        print(f"Fold {fold}")
        print(f"{'=' * 50}")
        for split in ["train", "calib", "test"]:
            ds = DCASE18Dataset(split=split, fold=fold, max_duration=1, use_cache=False)
            print(f"  {split}: {len(ds)} segments, {ds.num_classes} classes")
            # Print class distribution
            label_counts = {}
            for rec in ds.records:
                lbl = rec[1]
                label_counts[lbl] = label_counts.get(lbl, 0) + 1
            for lbl in sorted(label_counts.keys()):
                print(f"    {lbl}: {label_counts[lbl]} files")
