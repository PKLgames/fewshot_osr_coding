"""
YAMNet log-mel spectrogram extraction with GPU acceleration.

All computations are performed on GPU using PyTorch for maximum performance.
Input: torch.Tensor (batch_size, length) or (length,)
Output: torch.Tensor (batch_size, N, patch_frames, n_mels) or (N, patch_frames, n_mels)
"""
import torch
import numpy as np
import librosa


# ===============================
# Cache for mel filterbank and window
# ===============================
_mel_filterbank_cache = {}
_window_cache = {}


def _get_mel_filterbank(sample_rate: int, n_fft: int, n_mels: int, 
                        fmin: float, fmax: float, device: torch.device) -> torch.Tensor:
    """Get or create a mel filterbank matrix. Uses caching to avoid repeated computation."""
    cache_key = (sample_rate, n_fft, n_mels, fmin, fmax)
    
    if cache_key not in _mel_filterbank_cache:
        mel_basis = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels, 
                                        fmin=fmin, fmax=fmax)
        _mel_filterbank_cache[cache_key] = mel_basis.astype(np.float32)
    
    return torch.from_numpy(_mel_filterbank_cache[cache_key]).to(device)


def _get_hann_window(win_length: int, device: torch.device) -> torch.Tensor:
    """Get or create a Hann window. Uses caching to avoid repeated computation."""
    if win_length not in _window_cache:
        _window_cache[win_length] = torch.hann_window(win_length)
    
    return _window_cache[win_length].to(device)


def waveform_to_log_mel_patches(waveform: torch.Tensor,
                                sample_rate: int = 16000,
                                n_mels: int = 64,
                                window_seconds: float = 0.025,
                                hop_seconds: float = 0.01,
                                patch_frames: int = 96,
                                fmin: float = 0.0,
                                fmax: float = 8000.0) -> torch.Tensor:
    """
    Convert waveform to log-mel spectrogram patches (GPU accelerated).
    
    Args:
        waveform: Audio tensor (length,) or (batch_size, length) on GPU
        sample_rate: Sample rate (default: 16000)
        n_mels: Number of mel bands (default: 64)
        window_seconds: STFT window in seconds (default: 0.025)
        hop_seconds: STFT hop in seconds (default: 0.01)
        patch_frames: Frames per patch (default: 96)
        fmin: Minimum frequency (default: 0.0)
        fmax: Maximum frequency (default: 8000.0)
    
    Returns:
        Log-mel patches: (N, patch_frames, n_mels) or (batch_size, N, patch_frames, n_mels)
    """
    # Handle single waveform
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    batch_size = waveform.shape[0]
    device = waveform.device
    
    # STFT parameters
    n_fft = int(round(window_seconds * sample_rate))
    hop_length = int(round(hop_seconds * sample_rate))
    
    # Get cached filterbank and window
    mel_basis = _get_mel_filterbank(sample_rate, n_fft, n_mels, fmin, fmax, device)
    window = _get_hann_window(n_fft, device)
    
    # Convert to float
    waveform = waveform.float()
    
    # Compute STFT for entire batch (vectorized)
    spec = torch.stft(
        waveform, n_fft=n_fft, hop_length=hop_length, win_length=n_fft,
        window=window, center=True, return_complex=True
    )  # (batch_size, n_fft//2+1, frames)
    
    # Power spectrogram
    spec = torch.abs(spec) ** 2
    
    # Apply mel filterbank via einsum
    mel = torch.einsum('mf,bft->bmt', mel_basis, spec)  # (batch_size, n_mels, frames)
    
    # Log scale
    log_mel = torch.log(mel + 1e-8)
    
    # Transpose to (batch_size, frames, n_mels)
    log_mel = log_mel.permute(0, 2, 1)
    
    # Build patches (vectorized)
    n_frames = log_mel.shape[1]
    
    if n_frames < patch_frames:
        min_vals = log_mel.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
        pad = min_vals.expand(-1, patch_frames - n_frames, n_mels)
        log_mel = torch.cat([log_mel, pad], dim=1)
        n_frames = log_mel.shape[1]
    
    n_patches = n_frames // patch_frames
    
    if n_patches > 0:
        log_mel = log_mel[:, :n_patches * patch_frames, :]
        result = log_mel.reshape(batch_size, n_patches, patch_frames, n_mels)
    else:
        result = torch.zeros((batch_size, 1, patch_frames, n_mels), 
                            dtype=torch.float32, device=device)
    
    if squeeze_output:
        result = result.squeeze(0)
    
    return result