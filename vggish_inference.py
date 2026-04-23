"""
VGGish inference module for PyTorch with GPU support.

This module provides functions to extract VGGish embeddings from audio waveforms
using GPU acceleration.
"""

import torch
import torch.nn as nn
import numpy as np
import librosa


def waveform_to_log_mel_patches_vggish_gpu(waveform: torch.Tensor,
                                           sample_rate: int = 16000,
                                           n_mels: int = 64,
                                           window_seconds: float = 0.025,
                                           hop_seconds: float = 0.01,
                                           patch_frames: int = 96,
                                           fmin: float = 125.0,
                                           fmax: float = 7500.0) -> torch.Tensor:
    """
    GPU-based conversion of waveform to log-mel spectrogram patches compatible with VGGish.
    
    Args:
        waveform: Audio waveform tensor [batch, samples] or [samples] on GPU
        sample_rate: Sample rate (default: 16000)
        n_mels: Number of mel bands (default: 64)
        window_seconds: STFT window length in seconds (default: 0.025)
        hop_seconds: STFT hop length in seconds (default: 0.01)
        patch_frames: Number of frames per patch (default: 96)
        fmin: Minimum frequency (default: 125)
        fmax: Maximum frequency (default: 7500)
    
    Returns:
        Log-mel patches tensor [batch, n_patches, 96, 64] on GPU
    """
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    
    batch_size = waveform.shape[0]
    device = waveform.device
    
    # Parameters
    n_fft = int(round(window_seconds * sample_rate))
    hop_length = int(round(hop_seconds * sample_rate))
    win_length = n_fft
    
    # Create mel filterbank on GPU
    mel_basis = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
    mel_basis = torch.from_numpy(mel_basis).float().to(device)
    
    # Hann window on GPU
    window = torch.hann_window(win_length).to(device)
    
    all_patches = []
    
    for b in range(batch_size):
        wf = waveform[b].float()
        
        # Compute mel spectrogram using torch.stft on GPU
        spec = torch.stft(
            wf,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=True,
            return_complex=True
        )
        
        # Compute power spectrogram
        spec = torch.abs(spec) ** 2
        
        # Apply mel filterbank
        mel = torch.matmul(mel_basis, spec)
        
        # Convert to log scale (VGGish uses log_offset=0.01)
        log_mel = torch.log(mel + 0.01)
        
        # Transpose to (frames, n_mels)
        log_mel = log_mel.T
        
        # Build patches (VGGish uses 0.96s hop = 96 frames, no overlap)
        n_frames = log_mel.shape[0]
        if n_frames < patch_frames:
            # Pad with minimum value
            pad_amount = patch_frames - n_frames
            min_val = log_mel.min()
            pad = torch.full((pad_amount, n_mels), min_val, device=device)
            log_mel = torch.cat([log_mel, pad], dim=0)
            n_frames = log_mel.shape[0]
        
        # Number of non-overlapping patches
        n_patches = n_frames // patch_frames
        patches = []
        for i in range(n_patches):
            s = i * patch_frames
            patch = log_mel[s:s + patch_frames]
            patches.append(patch)
        
        if patches:
            patches = torch.stack(patches, dim=0)
        else:
            patches = torch.zeros((1, patch_frames, n_mels), dtype=torch.float32, device=device)
        
        all_patches.append(patches)
    
    # Pad to same number of patches across batch if needed
    max_patches = max(p.shape[0] for p in all_patches)
    for i in range(batch_size):
        if all_patches[i].shape[0] < max_patches:
            pad_size = max_patches - all_patches[i].shape[0]
            pad = torch.zeros((pad_size, patch_frames, n_mels), dtype=torch.float32, device=device)
            all_patches[i] = torch.cat([all_patches[i], pad], dim=0)
    
    result = torch.stack(all_patches, dim=0)
    return result


def waveform_to_log_mel_patches_gpu(waveform: torch.Tensor,
                                     sample_rate: int = 16000) -> torch.Tensor:
    """
    GPU-based version for compatibility with existing code.
    
    Args:
        waveform: Audio waveform [batch, samples] on GPU
        sample_rate: Sample rate (default: 16000)
    
    Returns:
        Log-mel patches [batch, n_patches, 96, 64] on GPU
    """
    return waveform_to_log_mel_patches_vggish_gpu(waveform, sample_rate=sample_rate)
