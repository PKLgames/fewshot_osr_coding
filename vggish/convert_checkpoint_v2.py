#!/usr/bin/env python
"""Convert TensorFlow VGGish checkpoint to PyTorch format."""

import sys
import numpy as np
import torch

# TensorFlow checkpoint reader
try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
except ImportError:
    print("TensorFlow not installed.")
    sys.exit(1)


def convert_vggish_checkpoint(tf_checkpoint_path, output_path):
    """Convert TensorFlow VGGish checkpoint to PyTorch format."""
    
    # Read TensorFlow checkpoint
    reader = tf.train.NewCheckpointReader(tf_checkpoint_path)
    
    # Get all variables
    var_to_shape_map = reader.get_variable_to_shape_map()
    
    # Create state dict
    state_dict = {}
    
    # Mapping from TensorFlow to PyTorch parameter names
    # Based on actual variable names in the checkpoint
    tf_to_pt_mapping = {
        # Conv layers
        'vggish/conv1/weights': ('conv1.weight', (3, 3, 1, 64)),
        'vggish/conv1/biases': ('conv1.bias', (64,)),
        'vggish/conv2/weights': ('conv2.weight', (3, 3, 64, 128)),
        'vggish/conv2/biases': ('conv2.bias', (128,)),
        # Conv3 (two conv layers)
        'vggish/conv3/conv3_1/weights': ('conv3.weight', (3, 3, 128, 256)),
        'vggish/conv3/conv3_1/biases': ('conv3.bias', (256,)),
        'vggish/conv3/conv3_2/weights': ('conv4.weight', (3, 3, 256, 256)),
        'vggish/conv3/conv3_2/biases': ('conv4.bias', (256,)),
        # Conv4 (two conv layers)
        'vggish/conv4/conv4_1/weights': ('conv5.weight', (3, 3, 256, 512)),
        'vggish/conv4/conv4_1/biases': ('conv5.bias', (512,)),
        'vggish/conv4/conv4_2/weights': ('conv6.weight', (3, 3, 512, 512)),
        'vggish/conv4/conv4_2/biases': ('conv6.bias', (512,)),
        # FC1 (two fc layers)
        'vggish/fc1/fc1_1/weights': ('fc1.weight', (4096, 12288)),
        'vggish/fc1/fc1_1/biases': ('fc1.bias', (4096,)),
        'vggish/fc1/fc1_2/weights': ('fc2.weight', (4096, 4096)),
        'vggish/fc1/fc1_2/biases': ('fc2.bias', (4096,)),
        # FC2 / embedding
        'vggish/fc2/weights': ('fc3.weight', (128, 4096)),
        'vggish/fc2/biases': ('fc3.bias', (128,)),
    }
    
    print("Converting checkpoint...")
    
    for tf_name, (pt_name, expected_shape) in tf_to_pt_mapping.items():
        if tf_name in var_to_shape_map:
            tensor = reader.get_tensor(tf_name)
            
            # Convert to PyTorch format
            # TensorFlow Conv2d weights: [height, width, in_channels, out_channels]
            # PyTorch Conv2d weights: [out_channels, in_channels, height, width]
            if 'weights' in tf_name and 'fc' not in tf_name:
                tensor = np.transpose(tensor, (3, 2, 0, 1))
            elif 'weights' in tf_name and 'fc' in tf_name:
                # FC weights: transpose for PyTorch
                tensor = tensor.T
            
            # Check shape
            actual_shape = tensor.shape
            if actual_shape != expected_shape:
                print(f"  Warning: {tf_name} -> {pt_name}: expected {expected_shape}, got {actual_shape}")
            
            state_dict[pt_name] = torch.from_numpy(tensor).float()
            print(f"  Converted: {tf_name} -> {pt_name} {tensor.shape}")
        else:
            print(f"  Warning: {tf_name} not found in checkpoint")
    
    # Save PyTorch checkpoint
    torch.save(state_dict, output_path)
    print(f"\nSaved PyTorch checkpoint to: {output_path}")
    print(f"Total parameters: {len(state_dict)}")
    
    return state_dict


def main():
    tf_checkpoint_path = 'vggish/vggish_model.ckpt'
    output_path = 'vggish/vggish_pytorch.pth'
    
    convert_vggish_checkpoint(tf_checkpoint_path, output_path)


if __name__ == '__main__':
    main()
