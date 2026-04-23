import torch
import numpy as np

def check_for_anomalies(tensor: torch.Tensor, name: str = "tensor") -> None:
    """
    检查张量中是否包含NaN、Inf或-Inf异常值
    
    参数:
    ----------
    tensor : torch.Tensor
        要检查的张量
    name : str
        张量的名称（用于错误信息）
    
    异常:
    ----------
    ValueError: 如果检测到异常值
    """
    # 检查NaN
    if torch.isnan(tensor).any():
        nan_count = torch.isnan(tensor).sum().item()
        raise ValueError(f"Error: {name} contains NaN values! Count: {nan_count}")
    
    # 检查正无穷
    if torch.isinf(tensor).any():
        inf_count = torch.isinf(tensor).sum().item()
        raise ValueError(f"Error: {name} contains Inf values! Count: {inf_count}")
    
    # 检查负无穷
    if torch.isinf(tensor).any():
        neg_inf_count = torch.sum(tensor == float('-inf')).item()
        if neg_inf_count > 0:
            raise ValueError(f"Error: {name} contains -Inf values! Count: {neg_inf_count}")
    
    # 如果没有异常，打印成功信息（可选）
    print(f"Success: {name} passed anomaly check (no NaN, Inf, or -Inf)")

# 使用示例
if __name__ == "__main__":
    # 创建正常张量
    normal_tensor = torch.randn(10, 5)
    
    # 创建包含异常值的张量
    problematic_tensor = torch.randn(10, 5)
    problematic_tensor[0, 0] = float('nan')  # 添加NaN
    problematic_tensor[1, 1] = float('inf')   # 添加Inf
    problematic_tensor[2, 2] = float('-inf')  # 添加-Inf
    
    # 测试正常张量
    try:
        check_for_anomalies(normal_tensor, "Normal Tensor")
        print("Normal tensor check passed!")
    except ValueError as e:
        print(f"Unexpected error: {e}")
    
    # 测试异常张量
    try:
        check_for_anomalies(problematic_tensor, "Problematic Tensor")
    except ValueError as e:
        print(f"Caught error: {e}")