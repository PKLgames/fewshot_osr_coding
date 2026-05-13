import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import librosa
import pandas as pd
from tqdm import tqdm


# 9-class label mapping for DCASE 2018 Task 5
# Base classes (seen during training): 0-4
# Novel classes (only in calib/test): 5-8
LABEL_TO_IX = {
    'absence': 0,
    'cooking': 1,
    'social_activity': 2,
    'watching_tv': 3,
    'working': 4,
    'dishwashing': 5,
    'eating': 6,
    'other': 7,
    'vacuum_cleaner': 8,
}


class DCASE18Pretrain(Dataset):
    """DCASE18 pretraining dataset, compatible with FOAC-AIFP Backbone pretraining.
    Returns (audio_waveform_1d, integer_label) per sample.
    """

    def __init__(self, root, phase='train', index=None, k=5, base_sess=None,
                 data_type='audio', args=None):
        self.root = root
        self.data_type = data_type
        self.phase = phase
        self.fold = getattr(args, 'fold', 1) if args is not None else 1

        csv_dir = os.path.join(root, 'sampled_setup')
        self.all_train_df = pd.read_csv(
            os.path.join(csv_dir, f'fold{self.fold}_train.csv'), sep='\t')
        self.all_calib_df = pd.read_csv(
            os.path.join(csv_dir, f'fold{self.fold}_calib.csv'), sep='\t')
        self.all_test_df = pd.read_csv(
            os.path.join(csv_dir, f'fold{self.fold}_evaluate.csv'), sep='\t')

        index = np.arange(index)

        if phase == 'train':
            self.data, self.targets = self._select_from_classes(
                self.all_train_df, index)
        elif phase == 'calib':
            self.data, self.targets = self._select_from_classes(
                self.all_calib_df, index)
        elif phase == 'test':
            self.data, self.targets = self._select_from_classes(
                self.all_test_df, index)

        # DCASE18 audio is 10s @ 16kHz = 160000 samples; we use 1s segments
        target_len = 16000
        print(f'Preloading {len(self.data)} audio samples into memory...')
        self.audio_cache = []
        self.segments_per_file = 10  # 10s / 1s
        for path in tqdm(self.data, desc='Caching audio'):
            audio, _ = librosa.load(path, sr=16000, mono=True)
            t = torch.tensor(audio, dtype=torch.float32)
            # Segment 10s audio into 10 x 1s chunks, keep all
            for seg_start in range(0, len(t), target_len):
                seg = t[seg_start:seg_start + target_len]
                if seg.shape[0] < target_len:
                    seg = F.pad(seg, (0, target_len - seg.shape[0]))
                self.audio_cache.append(seg)
        # Expand targets to match segments
        self.targets = np.repeat(self.targets, self.segments_per_file)
        print('Done.')

    def _select_from_classes(self, df, index):
        data_tmp = []
        targets_tmp = []
        for i in index:
            label_name = [k for k, v in LABEL_TO_IX.items() if v == i][0]
            ind_cl = np.where(df['scene_label'] == label_name)[0]
            for j in ind_cl:
                path = os.path.join(self.root, df['filename'].iloc[j])
                data_tmp.append(path)
                targets_tmp.append(i)
        return data_tmp, targets_tmp

    def __len__(self):
        return len(self.audio_cache)

    def __getitem__(self, i):
        return self.audio_cache[i], self.targets[i]


class OpenDCASE18(Dataset):
    """DCASE18 episode dataset for FOAC-AIFP few-shot open-set meta-train/test.

    Each __getitem__(episode_idx) returns a 10-tuple:
        support_xs, support_ys, query_xs, query_ys,
        suppopen_xs, suppopen_ys, openset_xs, openset_ys,
        base_ids, cls_open_ids
    matching the exact format used by OpenTAU22 / OpenTAU19.
    """

    def __init__(self, args, index, root, partition='test', fix_seed=True):
        super().__init__()
        self.fix_seed = fix_seed
        self.fold = getattr(args, 'fold', 1)
        if partition == 'test' and hasattr(args, 'test_n_ways') and args.test_n_ways is not None:
            self.n_ways = args.test_n_ways
            self.n_open_ways = args.test_n_open_ways
        else:
            self.n_ways = args.n_ways
            self.n_open_ways = args.n_open_ways
        self.n_shots = args.n_shots
        self.n_queries = args.n_queries
        self.n_episodes = (args.n_test_runs if partition in ('test', 'calib')
                           else args.n_train_runs)
        self.index = index
        self.root = root
        self.partition = partition
        self.train_classes = args.train_classes

        csv_dir = os.path.join(root, 'sampled_setup')
        self.all_train_df = pd.read_csv(
            os.path.join(csv_dir, f'fold{self.fold}_train.csv'), sep='\t')
        self.all_calib_df = pd.read_csv(
            os.path.join(csv_dir, f'fold{self.fold}_calib.csv'), sep='\t')
        self.all_test_df = pd.read_csv(
            os.path.join(csv_dir, f'fold{self.fold}_evaluate.csv'), sep='\t')

        # Build per-class file list
        self.data = {}
        for class_id in index:
            label_name = [k for k, v in LABEL_TO_IX.items() if v == class_id][0]

            if self.partition == 'train':
                df = self.all_train_df
                if label_name not in df['scene_label'].values:
                    continue
            elif self.partition == 'calib':
                df = self.all_calib_df
            else:
                df = self.all_test_df

            ind_cl = np.where(df['scene_label'] == label_name)[0]
            for j in ind_cl:
                path = os.path.join(self.root, df['filename'].iloc[j])
                self.data.setdefault(class_id, []).append(path)

        # Preload and segment audio (10s → 10 x 1s per file)
        target_len = 16000
        all_paths = set(p for paths in self.data.values() for p in paths)
        print(f'Preloading {len(all_paths)} unique audio files into memory...')
        self._audio_cache = {}  # path → list of 1s segments
        for p in tqdm(all_paths, desc='Caching audio'):
            audio, _ = librosa.load(p, sr=16000, mono=True)
            t = torch.tensor(audio, dtype=torch.float32)
            segments = []
            for seg_start in range(0, len(t), target_len):
                seg = t[seg_start:seg_start + target_len]
                if seg.shape[0] < target_len:
                    seg = F.pad(seg, (0, target_len - seg.shape[0]))
                segments.append(seg)
            self._audio_cache[p] = segments
        print('Done.')

        # Build segment pool: class_id → list of (path, seg_idx) tuples
        self._segment_pool = {}
        for class_id, paths in self.data.items():
            pool = []
            for p in paths:
                for seg_idx in range(len(self._audio_cache[p])):
                    pool.append((p, seg_idx))
            self._segment_pool[class_id] = pool

    # ------------------------------------------------------------------
    def _select_from_classes(self, df, index):
        data_tmp = []
        targets_tmp = []
        for i in index:
            label_name = [k for k, v in LABEL_TO_IX.items() if v == i][0]
            ind_cl = np.where(df['scene_label'] == label_name)[0]
            for j in ind_cl:
                path = os.path.join(self.root, df['filename'].iloc[j])
                data_tmp.append(path)
                targets_tmp.append(i)
        return data_tmp, targets_tmp

    # ------------------------------------------------------------------
    def __len__(self):
        return self.n_episodes

    def __getitem__(self, item):
        return self.get_episode(item)

    # ------------------------------------------------------------------
    def get_episode(self, item):
        if self.fix_seed:
            np.random.seed(item)

        available_classes = np.array(sorted(self._segment_pool.keys()))

        if self.partition in ('test', 'calib'):
            base_class_index = np.arange(self.train_classes)
            available_base = np.intersect1d(available_classes, base_class_index)
            available_novel = np.setxor1d(available_classes, available_base)

            cls_sampled = np.random.choice(available_base,
                min(self.n_ways, len(available_base)), False)
            cls_open_ids = np.random.choice(available_novel,
                min(self.n_open_ways, len(available_novel)), False)
        else:
            base_classes = available_classes

            max_closed = max(1, len(base_classes) - 1)
            n_ways_eff = min(self.n_ways, max_closed)
            n_open_eff = min(self.n_open_ways, len(base_classes) - n_ways_eff)

            perm = np.random.permutation(base_classes)
            cls_sampled = perm[:n_ways_eff]
            cls_open_ids = perm[n_ways_eff:n_ways_eff + n_open_eff]

        support_xs, support_ys = [], []
        query_xs, query_ys = [], []
        suppopen_xs, suppopen_ys = [], []
        openset_xs, openset_ys = [], []

        def _get_segment(p, seg_idx):
            return self._audio_cache[p][seg_idx]

        # ---- Closed-set: support + query ----
        for idx, the_cls in enumerate(cls_sampled):
            pool = self._segment_pool[the_cls]
            support_ids = np.random.choice(len(pool), self.n_shots, False)
            support_xs.extend([_get_segment(*pool[i]).view(1, -1) for i in support_ids])
            support_ys.extend([idx] * self.n_shots)

            query_pool = np.setxor1d(np.arange(len(pool)), support_ids)
            query_ids = np.random.choice(query_pool, self.n_queries, False)
            query_xs.extend([_get_segment(*pool[i]).view(1, -1) for i in query_ids])
            query_ys.extend([idx] * self.n_queries)

        support_xs = torch.cat(support_xs, dim=0)
        query_xs = torch.cat(query_xs, dim=0)

        # ---- Open-set: support-open + open query ----
        for idx, the_cls in enumerate(cls_open_ids):
            pool = self._segment_pool[the_cls]

            max_support = max(0, len(pool) - 1)
            actual_shots = min(self.n_shots, max_support)
            if actual_shots > 0:
                suppopen_ids = np.random.choice(len(pool), actual_shots, False)
            else:
                suppopen_ids = np.array([], dtype=int)
            suppopen_xs.extend([_get_segment(*pool[i]).view(1, -1) for i in suppopen_ids])
            suppopen_ys.extend([idx] * len(suppopen_ids))

            open_pool = np.setxor1d(np.arange(len(pool)), suppopen_ids)
            n_q = min(self.n_queries, len(open_pool))
            if n_q > 0:
                openset_ids = np.random.choice(open_pool, n_q, False)
            else:
                openset_ids = np.array([], dtype=int)
            openset_xs.extend([_get_segment(*pool[i]).view(1, -1) for i in openset_ids])
            openset_ys.extend([the_cls] * n_q)

        suppopen_xs = torch.cat(suppopen_xs, dim=0) if suppopen_xs else torch.empty(0)
        openset_xs = torch.cat(openset_xs, dim=0) if openset_xs else torch.empty(0)

        support_ys = np.array(support_ys)
        query_ys = np.array(query_ys)
        openset_ys = np.array(openset_ys)
        suppopen_ys = np.array(suppopen_ys)
        cls_sampled = np.array(cls_sampled)
        cls_open_ids = np.array(cls_open_ids)

        base_class_index = np.arange(self.train_classes)
        used_base = np.intersect1d(base_class_index,
                                   np.concatenate([cls_sampled, cls_open_ids]))
        base_ids = np.setxor1d(base_class_index, used_base)
        base_ids = np.array(sorted(base_ids))

        return (support_xs, support_ys, query_xs, query_ys,
                suppopen_xs, suppopen_ys, openset_xs, openset_ys,
                base_ids, cls_open_ids)
