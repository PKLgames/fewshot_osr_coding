"""
VGGish model implemented in PyTorch.

This is a PyTorch implementation of the VGGish audio embedding model.
Reference: https://github.com/tensorflow/models/tree/master/research/audioset/vggish
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VGGish(nn.Module):
    """
    VGGish model for audio embedding extraction.
    
    Input: Log-mel spectrogram patches [batch, n_patches, 96, 64]
    Output: 128-dimensional embeddings [batch, n_patches, 128] or [batch, 128]
    """
    
    def __init__(self, num_classes: int = 527):
        """
        Initialize VGGish model.
        
        Args:
            num_classes: Number of output classes (default: 527 for AudioSet)
                        Not used in embedding mode, but kept for compatibility.
        """
        super(VGGish, self).__init__()
        
        # Input shape: [batch, n_patches, 96, 64] or [batch, 1, 96, 64]
        # After reshaping: [batch*n_patches, 1, 96, 64]
        
        # Conv block 1: 96x64 -> 48x32
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Conv block 2: 48x32 -> 24x16
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Conv block 3: 24x16 -> 12x8
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Conv block 4: 12x8 -> 6x4
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # After 4 pooling layers: 96/16 = 6, 64/16 = 4
        # So input to fc is 512 * 6 * 4 = 12288
        
        # Fully connected layers
        self.fc1 = nn.Linear(512 * 6 * 4, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 128)  # Embedding layer (no activation)
        
        self.embedding_size = 128
        self.num_classes = num_classes
        
    def forward(self, x: torch.Tensor, to_prob: bool = False) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch, n_patches, 96, 64] or [batch, 1, 96, 64]
            to_prob: If True, apply softmax (not used in embedding mode)
        
        Returns:
            Embedding tensor of shape [batch, 128] (if single patch) or 
            [batch, n_patches, 128] (if multiple patches, takes first)
            or [batch, num_classes] if to_prob=True
        """
        original_shape = x.shape
        
        # Handle different input shapes
        if len(x.shape) == 3:
            # [batch, 96, 64] -> add channel dim
            x = x.unsqueeze(1)
        elif len(x.shape) == 4 and x.shape[1] != 1:
            # Multiple patches: [batch, n_patches, 96, 64]
            batch_size, num_patches = x.shape[:2]
            x = x.view(-1, 1, 96, 64)  # [batch*n_patches, 1, 96, 64]
        elif len(x.shape) == 4 and x.shape[1] == 1:
            # Single patch: [batch, 1, 96, 64]
            pass
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")
        
        # Conv block 1
        x = F.relu(self.conv1(x))
        x = self.pool1(x)  # [batch, 64, 48, 32]
        
        # Conv block 2
        x = F.relu(self.conv2(x))
        x = self.pool2(x)  # [batch, 128, 24, 16]
        
        # Conv block 3
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool3(x)  # [batch, 256, 12, 8]
        
        # Conv block 4
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool4(x)  # [batch, 512, 6, 4]
        
        # Flatten
        x = x.view(x.size(0), -1)  # [batch, 512*6*4]
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        embedding = self.fc3(x)  # No activation, pre-activation embedding [batch, 128]
        
        # Handle output based on original input
        if len(original_shape) == 4 and original_shape[1] != 1:
            # Multiple patches: reshape back and take mean
            embedding = embedding.view(batch_size, num_patches, -1)
            embedding = embedding.mean(dim=1)  # [batch, 128]
        
        if to_prob:
            # Apply softmax for classification (not typically used)
            embedding = F.softmax(embedding, dim=-1)
        
        return embedding
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Get embedding from input (alias for forward)."""
        return self.forward(x, to_prob=False)


def load_vggish_weights(model: nn.Module, checkpoint_path: str, device: str = 'cpu') -> nn.Module:
    """
    Load VGGish weights from a checkpoint file.
    
    Args:
        model: VGGish model instance
        checkpoint_path: Path to the checkpoint file
        device: Device to load weights to
    
    Returns:
        Model with loaded weights
    """
    try:
        # Try loading as PyTorch checkpoint
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded PyTorch checkpoint from {checkpoint_path}")
    except:
        print(f"Warning: Could not load checkpoint from {checkpoint_path}")
        print("Model will use random initialization.")
    
    return model


# Create a global instance for convenience
def get_vggish_model(pretrained: bool = False, pretrained_path: str = None, device: str = 'cuda'):
    """
    Get a VGGish model instance.
    
    Args:
        pretrained: If True, try to load pretrained weights
        pretrained_path: Path to pretrained weights (if None, uses default)
        device: Device to place model on
    
    Returns:
        VGGish model
    """
    model = VGGish()
    model = model.to(device)
    
    if pretrained and pretrained_path:
        model = load_vggish_weights(model, pretrained_path, device)
    
    model.eval()
    return model
