#!/usr/local/miniconda3/envs/py312/bin/python
"""
Audio Data Analysis Script - Raw Audio Sample Analysis Before Model Processing

Analysis Contents:
1. Dataset overview and class distribution
2. Waveform statistics (amplitude, energy, zero-crossing rate)
3. Frequency domain analysis (spectrum, spectral centroid, spectral bandwidth)
4. Log-Mel spectrogram analysis (actual model input)
5. Per-class feature statistics comparison
6. Inter-class similarity and t-SNE visualization
7. Data quality assessment

Output directory: experiment/TAU19_audio_analysis/
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.cm as cm
import os
import csv
import librosa
from collections import defaultdict, Counter
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from scipy.stats import gaussian_kde
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from utils.TAU22 import TAUDataset

# ============================================================
# Configuration
# ============================================================
OUTPUT_DIR = 'experiment/TAU22_1s_audio_analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Label mapping (consistent with vocabulary.csv)
LABEL_MAP = {
    0: 'airport',
    1: 'shopping_mall',
    2: 'metro_station',
    3: 'street_pedestrian',
    4: 'public_square',
    5: 'street_traffic',
    6: 'tram',
    7: 'bus',
    8: 'metro',
    9: 'park'
}

NUM_KNOWN = 6   # First 6 classes are known
SAMPLE_RATE = 16000
MAX_DURATION = 1  # seconds
NUM_CLASSES = 10

# VGGish parameters
VGGISH_N_MELS = 64
VGGISH_N_FFT = 400       # 0.025s * 16000
VGGISH_HOP_LENGTH = 160  # 0.01s * 16000
VGGISH_PATCH_FRAMES = 96

# Color mapping
CLASS_COLORS = plt.cm.tab10(np.linspace(0, 1, 10))

plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.1
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['axes.edgecolor'] = '#cccccc'


# ============================================================
# Utility Functions
# ============================================================
def load_dataset(split='train', max_samples=None):
    """Load preprocessed dataset via TAUDataset.
    Note: 'eval' split maps to 'test' in TAUDataset.
    Returns a TAUDataset instance (or list if max_samples set).
    Each item is a dict with 'source_audio', 'target', and 'idx' keys
    (compatible with original format).
    """
    # Map external split names to TAUDataset split names
    split_map = {
        'train': 'train',
        'calib': 'calib',
        'test': 'test',
        'eval': 'test',
    }
    split = split_map.get(split, split)
    print(f"  Loading TAUDataset(split='{split}', max_duration={MAX_DURATION}) ...")
    dataset = TAUDataset(split=split, max_duration=MAX_DURATION)
    if max_samples and len(dataset) > max_samples:
        # Build a lightweight wrapper to limit samples
        data = [dataset[i] for i in range(min(max_samples, len(dataset)))]
    else:
        data = dataset
    return data


def get_label_name(label_tensor):
    """Get class name from one-hot label tensor"""
    idx = label_tensor.squeeze().argmax().item()
    return LABEL_MAP.get(idx, f'class_{idx}'), idx


def compute_log_mel_spectrogram(waveform, sr=16000, n_mels=64, n_fft=400, hop_length=160):
    """Compute log-mel spectrogram"""
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.numpy()
    mel_spec = librosa.feature.melspectrogram(
        y=waveform, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels,
        fmin=125, fmax=7500, power=2.0
    )
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    return log_mel


def compute_spectral_features(waveform, sr=16000):
    """Compute frequency domain features"""
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.numpy()
    features = {}
    features['spectral_centroid'] = librosa.feature.spectral_centroid(y=waveform, sr=sr)[0].mean()
    features['spectral_bandwidth'] = librosa.feature.spectral_bandwidth(y=waveform, sr=sr)[0].mean()
    features['spectral_rolloff'] = librosa.feature.spectral_rolloff(y=waveform, sr=sr)[0].mean()
    features['spectral_flatness'] = librosa.feature.spectral_flatness(y=waveform).mean()
    features['zero_crossing_rate'] = librosa.feature.zero_crossing_rate(waveform)[0].mean()
    features['rms'] = librosa.feature.rms(y=waveform)[0].mean()
    mfccs = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f'mfcc_{i}'] = mfccs[i].mean()
    return features


# ============================================================
# Analysis 1: Dataset Overview and Class Distribution
# ============================================================
def analyze_class_distribution():
    """Analyze class distribution across dataset splits"""
    print("\n" + "="*70)
    print("Analysis 1: Dataset Overview and Class Distribution")
    print("="*70)
    
    splits = {'train': None, 'calib': None, 'eval': None}
    all_counts = {}
    
    for split_name in splits:
        data = load_dataset(split_name)
        counts = Counter()
        for item in data:
            _, idx = get_label_name(item['target'])
            counts[idx] += 1
        all_counts[split_name] = counts
        splits[split_name] = len(data)
        
        print(f"\n  [{split_name}] Total samples: {len(data)}")
        for cls_id in range(NUM_CLASSES):
            name = LABEL_MAP[cls_id]
            cnt = counts.get(cls_id, 0)
            role = "Known" if cls_id < NUM_KNOWN else "Unknown"
            print(f"    {cls_id:2d} {name:20s} ({role}): {cnt:6d} ({100*cnt/len(data):.1f}%)")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    split_names = ['train', 'calib', 'eval']
    
    for ax, split_name in zip(axes, split_names):
        counts = all_counts[split_name]
        classes = [LABEL_MAP[i] for i in range(NUM_CLASSES)]
        values = [counts.get(i, 0) for i in range(NUM_CLASSES)]
        colors = [CLASS_COLORS[i] for i in range(NUM_CLASSES)]
        
        bars = ax.bar(range(NUM_CLASSES), values, color=colors, edgecolor='none', linewidth=0)
        ax.set_xticks(range(NUM_CLASSES))
        ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Sample Count')
        ax.set_title(f'{split_name} (Total: {splits[split_name]})')
        
        ax.axvline(x=NUM_KNOWN - 0.5, color='red', linestyle='--', linewidth=1.5, label='Known/Unknown Boundary')
        ax.legend(fontsize=7)
        
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                       str(val), ha='center', va='bottom', fontsize=7)
    
    plt.suptitle('Class Distribution Comparison (Known vs Unknown)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/01_class_distribution.png', facecolor='white')
    plt.close()
    print(f"\n  [OK] Saved: {OUTPUT_DIR}/01_class_distribution.png")
    
    return all_counts


# ============================================================
# Analysis 2: Waveform Statistics
# ============================================================
def analyze_waveform_stats(max_samples_per_class=500):
    """Analyze waveform statistics per class"""
    print("\n" + "="*70)
    print("Analysis 2: Waveform Statistics")
    print("="*70)
    
    data = load_dataset('calib')
    
    class_waveforms = defaultdict(list)
    for item in data:
        _, idx = get_label_name(item['target'])
        if len(class_waveforms[idx]) < max_samples_per_class:
            class_waveforms[idx].append(item['source_audio'])
    
    stats_per_class = {}
    for cls_id in range(NUM_CLASSES):
        waveforms = class_waveforms[cls_id]
        amps = []
        rms_vals = []
        peak_vals = []
        zcr_vals = []
        
        for wf in waveforms:
            if isinstance(wf, torch.Tensor):
                wf_np = wf.numpy()
            else:
                wf_np = np.array(wf)
            amps.append(wf_np)
            rms_vals.append(np.sqrt(np.mean(wf_np**2)))
            peak_vals.append(np.max(np.abs(wf_np)))
            zcr_vals.append(np.sum(np.abs(np.diff(np.sign(wf_np))) > 0) / len(wf_np))
        
        all_amps = np.concatenate(amps)
        stats_per_class[cls_id] = {
            'mean_amp': np.mean(all_amps),
            'std_amp': np.std(all_amps),
            'min_amp': np.min(all_amps),
            'max_amp': np.max(all_amps),
            'rms_mean': np.mean(rms_vals),
            'rms_std': np.std(rms_vals),
            'peak_mean': np.mean(peak_vals),
            'peak_std': np.std(peak_vals),
            'zcr_mean': np.mean(zcr_vals),
            'zcr_std': np.std(zcr_vals),
        }
        
        name = LABEL_MAP[cls_id]
        role = "Known" if cls_id < NUM_KNOWN else "Unknown"
        s = stats_per_class[cls_id]
        print(f"  [{cls_id}] {name:20s} ({role}): "
              f"RMS={s['rms_mean']:.4f}+/-{s['rms_std']:.4f}, "
              f"Peak={s['peak_mean']:.4f}+/-{s['peak_std']:.4f}, "
              f"ZCR={s['zcr_mean']:.4f}+/-{s['zcr_std']:.4f}")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    rms_data = []
    for cls_id in range(NUM_CLASSES):
        rms_vals = [np.sqrt(np.mean(wf.numpy()**2)) for wf in class_waveforms[cls_id]]
        rms_data.append(rms_vals)
    
    bp1 = axes[0].boxplot(rms_data, patch_artist=True, labels=[LABEL_MAP[i][:6] for i in range(NUM_CLASSES)])
    for patch, color in zip(bp1['boxes'], CLASS_COLORS):
        patch.set_facecolor(color)
        patch.set_edgecolor('#666666')
        patch.set_linewidth(0.5)
    axes[0].axvline(x=NUM_KNOWN + 0.5, color='red', linestyle='--', linewidth=1.5)
    axes[0].set_ylabel('RMS Energy')
    axes[0].set_title('RMS Energy Distribution')
    axes[0].tick_params(axis='x', rotation=45)
    
    peak_data = []
    for cls_id in range(NUM_CLASSES):
        peak_vals = [np.max(np.abs(wf.numpy())) for wf in class_waveforms[cls_id]]
        peak_data.append(peak_vals)
    
    bp2 = axes[1].boxplot(peak_data, patch_artist=True, labels=[LABEL_MAP[i][:6] for i in range(NUM_CLASSES)])
    for patch, color in zip(bp2['boxes'], CLASS_COLORS):
        patch.set_facecolor(color)
        patch.set_edgecolor('#666666')
        patch.set_linewidth(0.5)
    axes[1].axvline(x=NUM_KNOWN + 0.5, color='red', linestyle='--', linewidth=1.5)
    axes[1].set_ylabel('Peak Amplitude')
    axes[1].set_title('Peak Amplitude Distribution')
    axes[1].tick_params(axis='x', rotation=45)
    
    zcr_data = []
    for cls_id in range(NUM_CLASSES):
        zcr_vals = []
        for wf in class_waveforms[cls_id]:
            wf_np = wf.numpy()
            zcr_vals.append(np.sum(np.abs(np.diff(np.sign(wf_np))) > 0) / len(wf_np))
        zcr_data.append(zcr_vals)
    
    bp3 = axes[2].boxplot(zcr_data, patch_artist=True, labels=[LABEL_MAP[i][:6] for i in range(NUM_CLASSES)])
    for patch, color in zip(bp3['boxes'], CLASS_COLORS):
        patch.set_facecolor(color)
        patch.set_edgecolor('#666666')
        patch.set_linewidth(0.5)
    axes[2].axvline(x=NUM_KNOWN + 0.5, color='red', linestyle='--', linewidth=1.5)
    axes[2].set_ylabel('Zero-Crossing Rate')
    axes[2].set_title('Zero-Crossing Rate Distribution')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.suptitle('Waveform Statistics Comparison (Red dashed: Known/Unknown boundary)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/02_waveform_stats.png', facecolor='white')
    plt.close()
    print(f"\n  [OK] Saved: {OUTPUT_DIR}/02_waveform_stats.png")
    
    return stats_per_class


# ============================================================
# Analysis 3: Sample Waveforms
# ============================================================
def plot_sample_waveforms(samples_per_class=3):
    """Plot typical waveforms and spectrograms per class"""
    print("\n" + "="*70)
    print("Analysis 3: Sample Waveforms and Spectrograms")
    print("="*70)
    
    data = load_dataset('calib')
    
    class_samples = defaultdict(list)
    for item in data:
        _, idx = get_label_name(item['target'])
        if len(class_samples[idx]) < samples_per_class:
            class_samples[idx].append(item['source_audio'])
    
    fig, axes = plt.subplots(NUM_CLASSES, samples_per_class * 2, figsize=(4 * samples_per_class * 2, 2 * NUM_CLASSES))
    
    for cls_id in range(NUM_CLASSES):
        for j, waveform in enumerate(class_samples[cls_id]):
            wf_np = waveform.numpy() if isinstance(waveform, torch.Tensor) else np.array(waveform)
            time_axis = np.arange(len(wf_np)) / SAMPLE_RATE
            
            ax_wave = axes[cls_id, j * 2] if samples_per_class > 1 else axes[cls_id]
            ax_wave.plot(time_axis, wf_np, color=CLASS_COLORS[cls_id], linewidth=0.5)
            ax_wave.set_xlim(0, MAX_DURATION)
            if j == 0:
                role = "Known" if cls_id < NUM_KNOWN else "Unknown"
                ax_wave.set_ylabel(f'{LABEL_MAP[cls_id]}\n({role})', fontsize=8, fontweight='bold')
            if cls_id == 0:
                ax_wave.set_title(f'Waveform #{j+1}', fontsize=9)
            ax_wave.tick_params(labelsize=6)
            
            ax_mel = axes[cls_id, j * 2 + 1] if samples_per_class > 1 else axes[cls_id]
            log_mel = compute_log_mel_spectrogram(wf_np)
            librosa.display.specshow(log_mel, sr=SAMPLE_RATE, hop_length=VGGISH_HOP_LENGTH,
                                     x_axis='time', y_axis='mel', ax=ax_mel, fmin=125, fmax=7500)
            if cls_id == 0:
                ax_mel.set_title(f'Log-Mel #{j+1}', fontsize=9)
            ax_mel.tick_params(labelsize=6)
    
    plt.suptitle('Typical Waveforms and Log-Mel Spectrograms per Class', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/03_waveform_mel_samples.png', facecolor='white')
    plt.close()
    print(f"  [OK] Saved: {OUTPUT_DIR}/03_waveform_mel_samples.png")


# ============================================================
# Analysis 4: Spectral Features
# ============================================================
def analyze_spectral_features(max_samples_per_class=300):
    """Analyze frequency domain features per class"""
    print("\n" + "="*70)
    print("Analysis 4: Spectral Feature Analysis")
    print("="*70)
    
    data = load_dataset('calib')
    
    class_samples = defaultdict(list)
    for item in data:
        _, idx = get_label_name(item['target'])
        if len(class_samples[idx]) < max_samples_per_class:
            class_samples[idx].append(item['source_audio'])
    
    all_features = defaultdict(lambda: defaultdict(list))
    
    for cls_id in range(NUM_CLASSES):
        print(f"  Computing spectral features for class {cls_id} ({LABEL_MAP[cls_id]})...")
        for wf in tqdm(class_samples[cls_id], desc=f'  {LABEL_MAP[cls_id]}'):
            wf_np = wf.numpy() if isinstance(wf, torch.Tensor) else np.array(wf)
            feat = compute_spectral_features(wf_np)
            for key, val in feat.items():
                all_features[cls_id][key].append(val)
    
    spectral_keys = ['spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff', 
                     'spectral_flatness', 'zero_crossing_rate', 'rms']
    
    print(f"\n  {'Class':20s} {'Centroid':>10s} {'Bandwidth':>10s} {'Rolloff':>10s} "
          f"{'Flatness':>10s} {'ZCR':>10s} {'RMS':>10s}")
    print("  " + "-" * 85)
    
    for cls_id in range(NUM_CLASSES):
        name = LABEL_MAP[cls_id]
        vals = [f"{np.mean(all_features[cls_id][k]):.4f}" for k in spectral_keys]
        print(f"  {name:20s} " + " ".join(f"{v:>10s}" for v in vals))
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    for ax_idx, key in enumerate(spectral_keys):
        row, col = ax_idx // 3, ax_idx % 3
        ax = axes[row, col]
        
        bp_data = [all_features[cls_id][key] for cls_id in range(NUM_CLASSES)]
        bp = ax.boxplot(bp_data, patch_artist=True, 
                       labels=[LABEL_MAP[i][:6] for i in range(NUM_CLASSES)])
        for patch, color in zip(bp['boxes'], CLASS_COLORS):
            patch.set_facecolor(color)
            patch.set_edgecolor('#666666')
            patch.set_linewidth(0.5)
        ax.axvline(x=NUM_KNOWN + 0.5, color='red', linestyle='--', linewidth=1.5)
        ax.set_title(key, fontsize=11, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Spectral Feature Distribution Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/04_spectral_features.png', facecolor='white')
    plt.close()
    print(f"\n  [OK] Saved: {OUTPUT_DIR}/04_spectral_features.png")
    
    return all_features


# ============================================================
# Analysis 5: Log-Mel Spectrogram Statistics
# ============================================================
def analyze_mel_statistics(max_samples_per_class=500):
    """Analyze Log-Mel spectrogram statistics (simulating model input)"""
    print("\n" + "="*70)
    print("Analysis 5: Log-Mel Spectrogram Statistics (VGGish Input)")
    print("="*70)
    
    data = load_dataset('calib')
    
    class_mel_stats = {}
    
    for cls_id in range(NUM_CLASSES):
        samples = []
        for item in data:
            _, idx = get_label_name(item['target'])
            if idx == cls_id and len(samples) < max_samples_per_class:
                samples.append(item['source_audio'])
        
        all_mels = []
        mel_means_per_band = defaultdict(list)
        
        for wf in tqdm(samples, desc=f'  {LABEL_MAP[cls_id]}'):
            wf_np = wf.numpy() if isinstance(wf, torch.Tensor) else np.array(wf)
            log_mel = compute_log_mel_spectrogram(wf_np)
            all_mels.append(log_mel)
            for band in range(log_mel.shape[0]):
                mel_means_per_band[band].append(log_mel[band].mean())
        
        all_mels_cat = np.concatenate([m.flatten() for m in all_mels])
        
        class_mel_stats[cls_id] = {
            'mean': np.mean(all_mels_cat),
            'std': np.std(all_mels_cat),
            'min': np.min(all_mels_cat),
            'max': np.max(all_mels_cat),
            'band_means': {b: np.mean(v) for b, v in mel_means_per_band.items()},
            'band_stds': {b: np.std(v) for b, v in mel_means_per_band.items()},
        }
        
        s = class_mel_stats[cls_id]
        print(f"  [{cls_id}] {LABEL_MAP[cls_id]:20s}: "
              f"mean={s['mean']:.2f}, std={s['std']:.2f}, "
              f"range=[{s['min']:.2f}, {s['max']:.2f}]")
    
    # Mel band energy distribution
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    
    for cls_id in range(NUM_CLASSES):
        row, col = cls_id // 5, cls_id % 5
        ax = axes[row, col]
        
        band_means = [class_mel_stats[cls_id]['band_means'].get(b, 0) for b in range(VGGISH_N_MELS)]
        band_stds = [class_mel_stats[cls_id]['band_stds'].get(b, 0) for b in range(VGGISH_N_MELS)]
        
        ax.plot(range(VGGISH_N_MELS), band_means, color=CLASS_COLORS[cls_id], linewidth=2)
        ax.fill_between(range(VGGISH_N_MELS),
                        np.array(band_means) - np.array(band_stds),
                        np.array(band_means) + np.array(band_stds),
                        color=CLASS_COLORS[cls_id], alpha=0.2)
        ax.set_title(f'{LABEL_MAP[cls_id]}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Mel Band')
        ax.set_ylabel('Mean dB')
        ax.grid(alpha=0.3)
        
        role = "Known" if cls_id < NUM_KNOWN else "Unknown"
        ax.text(0.95, 0.95, role, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat' if cls_id < NUM_KNOWN else 'lightblue', alpha=0.5))
    
    plt.suptitle('Mel Band Energy Distribution per Class (Mean +/- Std)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/05_mel_band_statistics.png', facecolor='white')
    plt.close()
    print(f"\n  [OK] Saved: {OUTPUT_DIR}/05_mel_band_statistics.png")
    
    # Average spectrogram comparison
    fig, axes = plt.subplots(2, 5, figsize=(25, 6))
    
    for cls_id in range(NUM_CLASSES):
        row, col = cls_id // 5, cls_id % 5
        ax = axes[row, col]
        
        samples = []
        count = 0
        for item in data:
            _, idx = get_label_name(item['target'])
            if idx == cls_id:
                samples.append(item['source_audio'])
                count += 1
                if count >= 200:
                    break
        
        mel_arrays = []
        for wf in samples:
            wf_np = wf.numpy() if isinstance(wf, torch.Tensor) else np.array(wf)
            log_mel = compute_log_mel_spectrogram(wf_np)
            mel_arrays.append(log_mel)
        
        min_frames = min(m.shape[1] for m in mel_arrays)
        mel_stack = np.stack([m[:, :min_frames] for m in mel_arrays])
        avg_mel = mel_stack.mean(axis=0)
        
        im = librosa.display.specshow(avg_mel, sr=SAMPLE_RATE, hop_length=VGGISH_HOP_LENGTH,
                                      x_axis='time', y_axis='mel', ax=ax, fmin=125, fmax=7500,
                                      cmap='viridis')
        ax.set_title(f'{LABEL_MAP[cls_id]}', fontsize=10, fontweight='bold')
    
    plt.suptitle('Average Log-Mel Spectrogram per Class', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/05_average_mel_spectrograms.png', facecolor='white')
    plt.close()
    print(f"  [OK] Saved: {OUTPUT_DIR}/05_average_mel_spectrograms.png")
    
    return class_mel_stats


# ============================================================
# Analysis 6: Feature Space Visualization (t-SNE + PCA)
# ============================================================
def analyze_feature_space(max_samples_per_class=300):
    """Visualize MFCC features using t-SNE and PCA"""
    print("\n" + "="*70)
    print("Analysis 6: Feature Space Visualization (t-SNE + PCA)")
    print("="*70)
    
    data = load_dataset('calib')
    
    features_list = []
    labels_list = []
    
    class_counts = defaultdict(int)
    
    print("  Extracting MFCC features...")
    for item in tqdm(data):
        _, idx = get_label_name(item['target'])
        if class_counts[idx] >= max_samples_per_class:
            continue
        class_counts[idx] += 1
        
        wf = item['source_audio']
        wf_np = wf.numpy() if isinstance(wf, torch.Tensor) else np.array(wf)
        
        mfcc = librosa.feature.mfcc(y=wf_np, sr=SAMPLE_RATE, n_mfcc=13)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        feat = np.concatenate([
            mfcc.mean(axis=1), mfcc.std(axis=1),
            mfcc_delta.mean(axis=1), mfcc_delta.std(axis=1),
            mfcc_delta2.mean(axis=1), mfcc_delta2.std(axis=1),
        ])
        
        features_list.append(feat)
        labels_list.append(idx)
    
    X = np.array(features_list)
    y = np.array(labels_list)
    
    print(f"  Feature matrix: {X.shape}, Labels: {len(y)}")
    
    print("  PCA reduction...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    print(f"  PCA explained variance ratio: {pca.explained_variance_ratio_}")
    
    print("  t-SNE reduction (may take a few minutes)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    X_tsne = tsne.fit_transform(X)
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    for cls_id in range(NUM_CLASSES):
        mask = y == cls_id
        name = LABEL_MAP[cls_id]
        role = "Known" if cls_id < NUM_KNOWN else "Unknown"
        marker = 'o' if cls_id < NUM_KNOWN else 's'
        label = f'{name} ({role})'
        
        axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1], 
                       c=[CLASS_COLORS[cls_id]], label=label, 
                       marker=marker, s=15, alpha=0.6, edgecolors='white', linewidth=0.3)
        axes[1].scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                       c=[CLASS_COLORS[cls_id]], label=label,
                       marker=marker, s=15, alpha=0.6, edgecolors='white', linewidth=0.3)
    
    axes[0].set_title('PCA (MFCC Features)', fontsize=13, fontweight='bold')
    axes[0].set_xlabel(f'PC1 ({100*pca.explained_variance_ratio_[0]:.1f}%)')
    axes[0].set_ylabel(f'PC2 ({100*pca.explained_variance_ratio_[1]:.1f}%)')
    axes[0].legend(fontsize=7, loc='best', ncol=2)
    axes[0].grid(alpha=0.3)
    
    axes[1].set_title('t-SNE (MFCC Features)', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('t-SNE 1')
    axes[1].set_ylabel('t-SNE 2')
    axes[1].legend(fontsize=7, loc='best', ncol=2)
    axes[1].grid(alpha=0.3)
    
    plt.suptitle('Audio Feature Space Visualization (o=Known, s=Unknown)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/06_feature_space_tsne_pca.png', facecolor='white')
    plt.close()
    print(f"\n  [OK] Saved: {OUTPUT_DIR}/06_feature_space_tsne_pca.png")
    
    return X, y


# ============================================================
# Analysis 7: Inter-Class Similarity
# ============================================================
def analyze_inter_class_similarity(max_samples_per_class=300):
    """Analyze feature similarity between classes"""
    print("\n" + "="*70)
    print("Analysis 7: Inter-Class Similarity Analysis")
    print("="*70)
    
    data = load_dataset('calib')
    
    class_features = defaultdict(list)
    
    for item in tqdm(data, desc='  Extracting features'):
        _, idx = get_label_name(item['target'])
        if len(class_features[idx]) >= max_samples_per_class:
            continue
        
        wf = item['source_audio']
        wf_np = wf.numpy() if isinstance(wf, torch.Tensor) else np.array(wf)
        mfcc = librosa.feature.mfcc(y=wf_np, sr=SAMPLE_RATE, n_mfcc=13)
        feat = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])
        class_features[idx].append(feat)
    
    class_means = np.array([np.mean(class_features[i], axis=0) for i in range(NUM_CLASSES)])
    dist_matrix = pairwise_distances(class_means, metric='cosine')
    sim_matrix = 1 - dist_matrix
    
    print("\n  Inter-class cosine similarity matrix:")
    header = "".join(f"{LABEL_MAP[i][:6]:>8s}" for i in range(NUM_CLASSES))
    print(f"  {'':8s}" + header)
    for i in range(NUM_CLASSES):
        row_vals = "".join(f"{sim_matrix[i, j]:8.3f}" for j in range(NUM_CLASSES))
        print(f"  {LABEL_MAP[i][:8]}{row_vals}")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(sim_matrix, cmap='RdYlBu_r', vmin=0, vmax=1)
    
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels([LABEL_MAP[i] for i in range(NUM_CLASSES)], rotation=45, ha='right')
    ax.set_yticklabels([LABEL_MAP[i] for i in range(NUM_CLASSES)])
    
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            text_color = 'white' if sim_matrix[i, j] > 0.7 else 'black'
            ax.text(j, i, f'{sim_matrix[i, j]:.2f}', ha='center', va='center',
                   fontsize=8, color=text_color, fontweight='bold')
    
    ax.axhline(y=NUM_KNOWN - 0.5, color='red', linewidth=2)
    ax.axvline(x=NUM_KNOWN - 0.5, color='red', linewidth=2)
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Cosine Similarity')
    ax.set_title('Inter-Class MFCC Cosine Similarity Matrix\n(Red lines: Known/Unknown boundary)', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/07_inter_class_similarity.png', facecolor='white')
    plt.close()
    print(f"\n  [OK] Saved: {OUTPUT_DIR}/07_inter_class_similarity.png")
    
    return sim_matrix


# ============================================================
# Analysis 8: Data Quality Assessment
# ============================================================
def analyze_data_quality(max_samples=5000):
    """Assess audio data quality"""
    print("\n" + "="*70)
    print("Analysis 8: Data Quality Assessment")
    print("="*70)
    
    data = load_dataset('train')
    total_data = len(data)
    sample_indices = np.random.choice(total_data, min(max_samples, total_data), replace=False)
    
    silence_count = 0
    clipping_count = 0
    very_quiet_count = 0
    very_loud_count = 0
    lengths = []
    
    for idx in tqdm(sample_indices, desc='  Quality check'):
        wf = data[idx]['source_audio']
        wf_np = wf.numpy() if isinstance(wf, torch.Tensor) else np.array(wf)
        lengths.append(len(wf_np))
        
        rms = np.sqrt(np.mean(wf_np**2))
        peak = np.max(np.abs(wf_np))
        
        if rms < 0.001:
            silence_count += 1
        if peak > 0.99:
            clipping_count += 1
        if rms < 0.01:
            very_quiet_count += 1
        if rms > 0.5:
            very_loud_count += 1
    
    total = len(sample_indices)
    print(f"\n  Samples checked: {total}")
    print(f"  Audio length: min={min(lengths)}, max={max(lengths)}, "
          f"expected={SAMPLE_RATE * MAX_DURATION}")
    print(f"  Silent samples (RMS<0.001): {silence_count} ({100*silence_count/total:.2f}%)")
    print(f"  Clipping samples (peak>0.99): {clipping_count} ({100*clipping_count/total:.2f}%)")
    print(f"  Very quiet samples (RMS<0.01): {very_quiet_count} ({100*very_quiet_count/total:.2f}%)")
    print(f"  Very loud samples (RMS>0.5): {very_loud_count} ({100*very_loud_count/total:.2f}%)")
    
    unique_lengths = Counter(lengths)
    print(f"  Length distribution: {dict(unique_lengths)}")
    
    all_amps = []
    class_amps = defaultdict(list)
    
    for idx in sample_indices[:2000]:
        wf = data[idx]['source_audio']
        _, cls_id = get_label_name(data[idx]['target'])
        wf_np = wf.numpy() if isinstance(wf, torch.Tensor) else np.array(wf)
        all_amps.extend(wf_np.tolist())
        class_amps[cls_id].extend(wf_np.tolist())
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(all_amps, bins=200, density=True, alpha=0.7, color='steelblue', edgecolor='none')
    axes[0].set_title('Global Amplitude Distribution', fontweight='bold')
    axes[0].set_xlabel('Amplitude')
    axes[0].set_ylabel('Probability Density')
    axes[0].set_yscale('log')
    axes[0].grid(alpha=0.3)
    
    for cls_id in range(NUM_CLASSES):
        if class_amps[cls_id]:
            axes[1].hist(class_amps[cls_id], bins=100, density=True, alpha=0.3,
                        color=CLASS_COLORS[cls_id], label=LABEL_MAP[cls_id])
    axes[1].set_title('Per-Class Amplitude Distribution', fontweight='bold')
    axes[1].set_xlabel('Amplitude')
    axes[1].set_ylabel('Probability Density')
    axes[1].set_yscale('log')
    axes[1].legend(fontsize=7, ncol=2)
    axes[1].grid(alpha=0.3)
    
    plt.suptitle('Data Quality Assessment - Amplitude Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/08_data_quality.png', facecolor='white')
    plt.close()
    print(f"\n  [OK] Saved: {OUTPUT_DIR}/08_data_quality.png")


# ============================================================
# Analysis 9: Known vs Unknown Feature Differences
# ============================================================
def analyze_known_vs_unknown(max_samples_per_class=300):
    """Compare feature differences between known and unknown classes"""
    print("\n" + "="*70)
    print("Analysis 9: Known vs Unknown Class Feature Differences")
    print("="*70)
    
    data = load_dataset('calib')
    
    known_features = []
    unknown_features = []
    
    class_counts = defaultdict(int)
    
    for item in tqdm(data, desc='  Extracting features'):
        _, idx = get_label_name(item['target'])
        if class_counts[idx] >= max_samples_per_class:
            continue
        class_counts[idx] += 1
        
        wf = item['source_audio']
        wf_np = wf.numpy() if isinstance(wf, torch.Tensor) else np.array(wf)
        
        feat_dict = compute_spectral_features(wf_np)
        feat_vec = [feat_dict[k] for k in sorted(feat_dict.keys())]
        
        if idx < NUM_KNOWN:
            known_features.append(feat_vec)
        else:
            unknown_features.append(feat_vec)
    
    known_features = np.array(known_features)
    unknown_features = np.array(unknown_features)
    
    feat_names = sorted(feat_dict.keys())
    
    print(f"\n  Known samples: {len(known_features)}, Unknown samples: {len(unknown_features)}")
    print(f"\n  {'Feature':25s} {'Known Mean':>10s} {'Unknown Mean':>10s} {'Diff':>10s} {'Cohen-d':>10s}")
    print("  " + "-" * 70)
    
    for i, name in enumerate(feat_names):
        k_mean = np.mean(known_features[:, i])
        u_mean = np.mean(unknown_features[:, i])
        diff = u_mean - k_mean
        pooled_std = np.sqrt((np.var(known_features[:, i]) + np.var(unknown_features[:, i])) / 2)
        cohens_d = diff / pooled_std if pooled_std > 0 else 0
        
        print(f"  {name:25s} {k_mean:10.4f} {u_mean:10.4f} {diff:10.4f} {cohens_d:10.4f}")
    
    key_features = ['spectral_centroid', 'spectral_bandwidth', 'spectral_flatness', 
                    'zero_crossing_rate', 'rms']
    
    fig, axes = plt.subplots(1, len(key_features), figsize=(5 * len(key_features), 5))
    
    for ax, feat_name in zip(axes, key_features):
        feat_idx = feat_names.index(feat_name)
        
        ax.hist(known_features[:, feat_idx], bins=50, alpha=0.5, density=True,
               color='coral', label=f'Known (0-{NUM_KNOWN-1})')
        ax.hist(unknown_features[:, feat_idx], bins=50, alpha=0.5, density=True,
               color='steelblue', label=f'Unknown ({NUM_KNOWN}-{NUM_CLASSES-1})')
        
        ax.set_title(feat_name, fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    
    plt.suptitle('Known vs Unknown Class - Key Feature Distribution Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/09_known_vs_unknown.png', facecolor='white')
    plt.close()
    print(f"\n  [OK] Saved: {OUTPUT_DIR}/09_known_vs_unknown.png")


# ============================================================
# Analysis 10: VGGish Processing Pipeline Simulation
# ============================================================
def analyze_vggish_pipeline(max_samples_per_class=100):
    """Simulate VGGish processing pipeline, show features the model actually sees"""
    print("\n" + "="*70)
    print("Analysis 10: VGGish Processing Pipeline Simulation")
    print("="*70)
    
    import yamnet_PT_inference as yamnet_infer
    
    data = load_dataset('calib')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    class_samples = defaultdict(list)
    for item in data:
        _, idx = get_label_name(item['target'])
        if len(class_samples[idx]) < max_samples_per_class:
            class_samples[idx].append(item['source_audio'])
    
    class_patch_stats = {}
    
    for cls_id in range(NUM_CLASSES):
        waveforms = class_samples[cls_id]
        batch = torch.stack(waveforms[:min(64, len(waveforms))]).to(device)
        
        with torch.no_grad():
            patches = yamnet_infer.waveform_to_log_mel_patches(batch, sample_rate=SAMPLE_RATE)
        
        patches_np = patches.cpu().numpy()
        
        class_patch_stats[cls_id] = {
            'shape': patches.shape,
            'mean': patches_np.mean(),
            'std': patches_np.std(),
            'min': patches_np.min(),
            'max': patches_np.max(),
            'n_patches': patches.shape[1],
            'samples': patches_np,
        }
        
        s = class_patch_stats[cls_id]
        print(f"  [{cls_id}] {LABEL_MAP[cls_id]:20s}: shape={list(s['shape'])}, "
              f"mean={s['mean']:.3f}, std={s['std']:.3f}, "
              f"range=[{s['min']:.3f}, {s['max']:.3f}], "
              f"n_patches={s['n_patches']}")
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    
    for cls_id in range(NUM_CLASSES):
        row, col = cls_id // 5, cls_id % 5
        ax = axes[row, col]
        
        patch = class_patch_stats[cls_id]['samples'][0, 0]
        im = ax.imshow(patch.T, aspect='auto', origin='lower', cmap='viridis',
                       vmin=-3, vmax=3)
        ax.set_title(f'{LABEL_MAP[cls_id]}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Time Frame')
        ax.set_ylabel('Mel Band')
        
        role = "Known" if cls_id < NUM_KNOWN else "Unknown"
        ax.text(0.95, 0.95, role, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat' if cls_id < NUM_KNOWN else 'lightblue', alpha=0.5))
    
    plt.suptitle('VGGish Input: Typical Log-Mel Patch [96x64]', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/10_vggish_patches.png', facecolor='white')
    plt.close()
    print(f"\n  [OK] Saved: {OUTPUT_DIR}/10_vggish_patches.png")


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("TAU Audio Data Analysis - Pre-Model Data Exploration")
    print("=" * 70)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Dataset: TAU-urban-acoustic-scenes-2022-mobile")
    print(f"  - Known classes (0-5): {', '.join(LABEL_MAP[i] for i in range(NUM_KNOWN))}")
    print(f"  - Unknown classes (6-9): {', '.join(LABEL_MAP[i] for i in range(NUM_KNOWN, NUM_CLASSES))}")
    print(f"  - Sample rate: {SAMPLE_RATE}Hz, Duration: {MAX_DURATION}s")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("Starting analysis...")
    print("=" * 70)
    
    counts = analyze_class_distribution()
    wf_stats = analyze_waveform_stats()
    plot_sample_waveforms(samples_per_class=2)
    spec_features = analyze_spectral_features()
    mel_stats = analyze_mel_statistics()
    X, y = analyze_feature_space()
    sim_matrix = analyze_inter_class_similarity()
    analyze_data_quality()
    analyze_known_vs_unknown()
    analyze_vggish_pipeline()
    
    print("\n" + "=" * 70)
    print("Analysis Complete! Generated files:")
    print("=" * 70)
    
    output_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png')])
    for f in output_files:
        size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
        print(f"  [chart] {f} ({size/1024:.1f} KB)")
    
    print(f"\nAll charts saved in: {OUTPUT_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
