import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import librosa
import pandas as pd
from tqdm import tqdm


# Label mapping (from TAU-19 vocabulary.csv)
LABEL_TO_IX = {
    'airport': 0, 'shopping_mall': 1, 'metro_station': 2,
    'street_pedestrian': 3, 'public_square': 4, 'street_traffic': 5,
    'tram': 6, 'bus': 7, 'metro': 8, 'park': 9,
}


class TAU19Pretrain(Dataset):
    """TAU-19 pretraining dataset, compatible with FOAC-AIFP Backbone pretraining.
    Returns (audio_waveform_1d, integer_label) per sample.
    """

    def __init__(self, root, phase='train', index=None, k=5, base_sess=None,
                 data_type='audio', args=None):
        self.root = root
        self.data_type = data_type
        self.phase = phase

        csv_dir = os.path.join(root, 'sampled_setup')
        self.all_train_df = pd.read_csv(
            os.path.join(csv_dir, 'sampled_fold1_train.csv'), sep='\t')
        self.all_calib_df = pd.read_csv(
            os.path.join(csv_dir, 'sampled_fold1_calib.csv'), sep='\t')
        self.all_test_df = pd.read_csv(
            os.path.join(csv_dir, 'sampled_fold1_evaluate.csv'), sep='\t')

        index = np.arange(index)  # e.g. np.arange(6) → [0,1,2,3,4,5]

        if phase == 'train':
            self.data, self.targets = self._select_from_classes(
                self.all_train_df, index)
        elif phase == 'calib':
            self.data, self.targets = self._select_from_classes(
                self.all_calib_df, index)
        elif phase == 'test':
            self.data, self.targets = self._select_from_classes(
                self.all_test_df, index)

        # Cache all audio in memory (fixed length: 16000 samples = 1s @ 16kHz)
        target_len = 16000
        print(f'Preloading {len(self.data)} audio samples into memory...')
        self.audio_cache = []
        for path in tqdm(self.data, desc='Caching audio'):
            audio, _ = librosa.load(path, sr=16000, mono=True)
            t = torch.tensor(audio, dtype=torch.float32)
            if t.shape[0] > target_len:
                t = t[:target_len]
            elif t.shape[0] < target_len:
                t = F.pad(t, (0, target_len - t.shape[0]))
            self.audio_cache.append(t)
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
        return len(self.data)

    def __getitem__(self, i):
        return self.audio_cache[i], self.targets[i]


class OpenTAU19(Dataset):
    """TAU-19 episode dataset for FOAC-AIFP few-shot open-set meta-train/test.

    Each __getitem__(episode_idx) returns a 10-tuple:
        support_xs, support_ys, query_xs, query_ys,
        suppopen_xs, suppopen_ys, openset_xs, openset_ys,
        base_ids, cls_open_ids
    matching the exact format used by Opennds / Openfmc / Openlbrs.
    """

    def __init__(self, args, index, root, partition='test', fix_seed=True):
        super().__init__()
        self.fix_seed = fix_seed
        # For test partition, use test-specific way settings if available
        if partition == 'test' and hasattr(args, 'test_n_ways') and args.test_n_ways is not None:
            self.n_ways = args.test_n_ways
            self.n_open_ways = args.test_n_open_ways
        else:
            self.n_ways = args.n_ways
            self.n_open_ways = args.n_open_ways
        self.n_shots = args.n_shots
        self.n_queries = args.n_queries
        self.n_episodes = (args.n_test_runs if partition == 'test'
                           else args.n_train_runs)
        self.index = index
        self.root = root
        self.partition = partition
        self.train_classes = args.train_classes

        csv_dir = os.path.join(root, 'sampled_setup')
        self.all_train_df = pd.read_csv(
            os.path.join(csv_dir, 'sampled_fold1_train.csv'), sep='\t')
        self.all_test_df = pd.read_csv(
            os.path.join(csv_dir, 'sampled_fold1_evaluate.csv'), sep='\t')

        # Build per-class file list: for train partition, load base from train CSV, novel from eval CSV
        self.data = {}
        for class_id in index:
            label_name = [k for k, v in LABEL_TO_IX.items() if v == class_id][0]

            # For train partition, try train CSV first, then eval CSV for novel classes
            if self.partition == 'train':
                # Check if class exists in train CSV (base classes)
                if label_name in self.all_train_df['scene_label'].values:
                    df = self.all_train_df
                else:
                    # Novel classes not in train CSV, load from eval CSV
                    df = self.all_test_df
            else:
                df = self.all_test_df

            ind_cl = np.where(df['scene_label'] == label_name)[0]
            for j in ind_cl:
                path = os.path.join(self.root, df['filename'].iloc[j])
                self.data.setdefault(class_id, []).append(path)

        # Preload all audio into memory (fixed length: 16000 samples = 1s @ 16kHz)
        target_len = 16000
        all_paths = set(p for paths in self.data.values() for p in paths)
        print(f'Preloading {len(all_paths)} unique audio files into memory...')
        self._audio_cache = {}
        for p in tqdm(all_paths, desc='Caching audio'):
            audio, _ = librosa.load(p, sr=16000, mono=True)
            t = torch.tensor(audio, dtype=torch.float32)
            if t.shape[0] > target_len:
                t = t[:target_len]
            elif t.shape[0] < target_len:
                t = F.pad(t, (0, target_len - t.shape[0]))
            self._audio_cache[p] = t
        print('Done.')

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

        # Available classes from the data we loaded
        available_classes = np.array(sorted(self.data.keys()))

        if self.partition == 'test':
            # Test partition: all classes are novel, split into known and open
            n_total_needed = self.n_ways + self.n_open_ways
            n_available = len(available_classes)
            if n_available >= n_total_needed:
                # Sample all at once then split
                sampled = np.random.choice(available_classes, n_total_needed, False)
                cls_sampled = sampled[:self.n_ways]
                cls_open_ids = sampled[self.n_ways:]
            else:
                # Not enough classes - split available classes
                n_known = min(self.n_ways, max(1, n_available - self.n_open_ways))
                n_open = min(self.n_open_ways, n_available - n_known)
                perm = np.random.permutation(available_classes)
                cls_sampled = perm[:n_known]
                cls_open_ids = perm[n_known:n_known + n_open]
        else:
            # Train partition: base classes [0-5] for known, novel [6-9] for open
            base_classes = np.intersect1d(np.arange(self.train_classes), available_classes)
            novel_classes = np.setxor1d(available_classes, base_classes)

            cls_sampled = np.random.choice(base_classes,
                min(self.n_ways, len(base_classes)), False)

            if len(novel_classes) >= self.n_open_ways:
                cls_open_ids = np.random.choice(novel_classes, self.n_open_ways, False)
            else:
                remaining_base = np.setxor1d(base_classes, cls_sampled)
                if len(remaining_base) >= self.n_open_ways:
                    cls_open_ids = np.random.choice(remaining_base, self.n_open_ways, False)
                else:
                    cls_open_ids = np.random.choice(available_classes, self.n_open_ways, False)

        support_xs, support_ys = [], []
        query_xs, query_ys = [], []
        suppopen_xs, suppopen_ys = [], []
        openset_xs, openset_ys = [], []

        # ---- Closed-set: support + query ----
        for idx, the_cls in enumerate(cls_sampled):
            paths = self.data[the_cls]
            audio = [self._audio_cache[p] for p in paths]

            support_ids = np.random.choice(len(audio), self.n_shots, False)
            support_xs.extend([audio[i].view(1, -1) for i in support_ids])
            support_ys.extend([idx] * self.n_shots)

            query_pool = np.setxor1d(np.arange(len(audio)), support_ids)
            query_ids = np.random.choice(query_pool, self.n_queries, False)
            query_xs.extend([audio[i].view(1, -1) for i in query_ids])
            query_ys.extend([idx] * self.n_queries)

        support_xs = torch.cat(support_xs, dim=0)
        query_xs = torch.cat(query_xs, dim=0)

        # ---- Open-set: support-open + open query ----
        for idx, the_cls in enumerate(cls_open_ids):
            paths = self.data[the_cls]
            audio = [self._audio_cache[p] for p in paths]

            suppopen_ids = np.random.choice(len(audio), self.n_shots, False)
            suppopen_xs.extend([audio[i].view(1, -1) for i in suppopen_ids])
            suppopen_ys.extend([idx] * self.n_shots)

            openset_ids = np.random.choice(len(audio), self.n_queries, False)
            openset_xs.extend([audio[i].view(1, -1) for i in openset_ids])
            openset_ys.extend([the_cls] * self.n_queries)

        suppopen_xs = torch.cat(suppopen_xs, dim=0)
        openset_xs = torch.cat(openset_xs, dim=0)

        support_ys = np.array(support_ys)
        query_ys = np.array(query_ys)
        openset_ys = np.array(openset_ys)
        suppopen_ys = np.array(suppopen_ys)
        cls_sampled = np.array(cls_sampled)
        cls_open_ids = np.array(cls_open_ids)

        # base_ids: indices into weight_base/weight_base_open (base class range only)
        base_class_index = np.arange(self.train_classes)
        used_base = np.intersect1d(base_class_index,
                                   np.concatenate([cls_sampled, cls_open_ids]))
        base_ids = np.setxor1d(base_class_index, used_base)
        base_ids = np.array(sorted(base_ids))

        return (support_xs, support_ys, query_xs, query_ys,
                suppopen_xs, suppopen_ys, openset_xs, openset_ys,
                base_ids, cls_open_ids)
