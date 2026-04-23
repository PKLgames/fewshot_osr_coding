import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
fm.fontManager = fm.FontManager()
from scipy.stats import multivariate_normal
import matplotlib.cm as cm
import os
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import kl_div
import random

# 设置中文字体支持
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子以便结果可重现
np.random.seed(42)

# 1. 定义参数
num_components = 3  # 高斯分布数量
input_dim = 2  # 输入维度

# 生成混合系数 π_k (归一化使得和为1)
pi = np.random.dirichlet(np.ones(num_components) * 1.5)

# 为每个高斯成分定义均值和协方差矩阵
# 标准高斯分布：期望为0，方差为I
means = []
covariances = []

# 定义三个标准高斯分布，但通过线性变换得到不同的位置
# 标准高斯分布：期望为0，方差为I
standard_mean = np.array([0, 0])
standard_cov = np.eye(2)  # 单位矩阵

# 但为了让混合分布有多个峰值，我们可以对标准高斯分布进行平移
# 这样每个成分仍然是高斯分布，但均值不同
means.append(np.array([2.0, 2.0]))
means.append(np.array([-2.0, 2.0]))
means.append(np.array([0, -2.0]))

# 所有成分的协方差矩阵都是单位矩阵（标准高斯分布）
for _ in range(num_components):
    covariances.append(np.eye(2))

# 2. 定义混合高斯分布的概率密度函数
def gaussian_mixture_pdf(x, means, covariances, weights):
    """
    计算混合高斯分布在点x处的概率密度
    
    参数:
    x: 输入点，形状为 (n_points, 2)
    means: 均值列表
    covariances: 协方差矩阵列表
    weights: 权重列表
    
    返回:
    概率密度值
    """
    n_points = x.shape[0]
    density = np.zeros(n_points)
    
    for i, (mean, cov, weight) in enumerate(zip(means, covariances, weights)):
        # 创建多元高斯分布对象
        mvn = multivariate_normal(mean=mean, cov=cov)
        density += random.uniform(0.9, 1.1) * weight * mvn.pdf(x)
    
    return density

# 3. 从混合高斯分布中采样
num_samples = 1000
samples = []

# 通过混合采样
for _ in range(num_samples):
    # 根据权重选择一个成分
    k = np.random.choice(num_components, p=pi)
    
    # 从选中的高斯分布中采样
    sample = np.random.multivariate_normal(means[k], covariances[k])
    samples.append(sample)

samples = np.array(samples)

# 4. 创建网格用于计算概率密度
x = np.linspace(-3, 3, 200)
y = np.linspace(-3, 3, 200)
X, Y = np.meshgrid(x, y)

# 将网格点展平以便批量计算
grid_points = np.column_stack([X.ravel(), Y.ravel()])

# 计算每个网格点的概率密度
Z = gaussian_mixture_pdf(grid_points, means, covariances, pi)
Z = Z.reshape(X.shape)

# 5. 绘制图像
plt.figure(figsize=(12, 10))

# 5.1 主图：散点图 + 概率密度等高线
plt.subplot(2, 2, 1)
# 绘制散点图
plt.scatter(samples[:, 0], samples[:, 1], c='red', s=10, alpha=0.7, label='采样点', edgecolor='black', linewidth=0.3)
# 绘制概率密度等高线
contour = plt.contour(X, Y, Z, levels=12, colors='black', linewidths=1.5, alpha=0.8)
plt.clabel(contour, inline=True, fontsize=8, colors='black')
plt.xlabel('X', fontsize=12)
plt.ylabel('Y', fontsize=12)
plt.title('标准高斯分布混合的散点图与概率密度等高线', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')

# 5.2 概率密度热力图
plt.subplot(2, 2, 2)
# 绘制概率密度热力图
heatmap = plt.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.8)
# 叠加散点
plt.scatter(samples[:, 1], samples[:, 0], c='red', s=5, alpha=0.5, label='采样点', edgecolor='black', linewidth=0.2)
plt.xlabel('X', fontsize=12)
plt.ylabel('Y', fontsize=12)
plt.title('概率密度热力图与采样点', fontsize=14)
plt.legend()
plt.colorbar(heatmap, label='概率密度')
plt.grid(True, alpha=0.3)
plt.axis('equal')

# 5.3 三维曲面图
from mpl_toolkits.mplot3d import Axes3D
ax = plt.subplot(2, 2, 3, projection='3d')
# 绘制概率密度曲面
surf = ax.plot_surface(X, Y, Z, cmap='viridis', 
                       alpha=0.8, linewidth=0, antialiased=True, rstride=2, cstride=2)
# 绘制散点（投影到XY平面）
ax.scatter(samples[:, 0], samples[:, 1], np.zeros(num_samples), 
           c='red', s=5, alpha=0.6, label='采样点', depthshade=True)
ax.set_xlabel('X', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.set_zlabel('概率密度', fontsize=12)
ax.set_title('概率密度三维曲面图', fontsize=14)
ax.legend()
plt.colorbar(surf, ax=ax, shrink=0.7, label='概率密度')

# 5.4 单独的高斯成分可视化
plt.subplot(2, 2, 4)
# 为每个高斯成分绘制等高线
colors = ['blue', 'green', 'orange']
for i, (mean, cov, weight, color) in enumerate(zip(means, covariances, pi, colors)):
    # 计算单个高斯分布的概率密度
    mvn = multivariate_normal(mean=mean, cov=cov)
    Z_single = mvn.pdf(grid_points).reshape(X.shape)
    
    # 绘制等高线
    contour_single = plt.contour(X, Y, Z_single, levels=5, colors=color, 
                                 linewidths=1.2, alpha=0.7, linestyles='--')
    plt.clabel(contour_single, inline=True, fontsize=7, colors=color)
    
    # 标记均值点
    plt.scatter(mean[0], mean[1], c=color, s=100, marker='*', edgecolor='black', 
                linewidth=1, alpha=0.9, label=f'成分{i+1} (权重: {weight:.2f})')

# 绘制混合分布的散点
plt.scatter(samples[:, 0], samples[:, 1], c='red', s=5, alpha=0.5, label='混合采样点')
plt.xlabel('X', fontsize=12)
plt.ylabel('Y', fontsize=12)
plt.title('各个标准高斯成分的等高线', fontsize=14)
plt.legend(loc='upper right', fontsize=9)
plt.grid(True, alpha=0.3)
plt.axis('equal')

plt.suptitle('三个标准高斯分布（不同均值）加权求和的概率密度可视化', fontsize=16, y=1.02)
plt.tight_layout()

# 6. 保存和显示图像
plt.savefig('standard_gaussian_mixture_scatter_contour.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# 7. 打印生成参数
print("=== 生成的混合高斯模型参数 ===")
print(f"混合成分数: {num_components}")
print(f"混合系数 (权重): {pi}")

for i, (mean, cov, weight) in enumerate(zip(means, covariances, pi)):
    print(f"\n--- 高斯成分 {i+1} ---")
    print(f"权重: {weight:.4f}")
    print(f"均值: {mean}")
    print(f"协方差矩阵:")
    print(cov)
    print(f"是否为单位矩阵: {np.allclose(cov, np.eye(2))}")

# 8. 统计信息
print("\n=== 采样统计信息 ===")
print(f"采样点数: {num_samples}")
print(f"采样点范围:")
print(f"  X: [{samples[:, 0].min():.3f}, {samples[:, 0].max():.3f}]")
print(f"  Y: [{samples[:, 1].min():.3f}, {samples[:, 1].max():.3f}]")
print(f"采样点均值: [{samples[:, 0].mean():.3f}, {samples[:, 1].mean():.3f}]")
print(f"采样点协方差矩阵:")
print(np.cov(samples, rowvar=False))

# 9. 验证每个成分是标准高斯分布
print("\n=== 标准高斯分布验证 ===")
for i, (mean, cov) in enumerate(zip(means, covariances)):
    print(f"\n成分 {i+1}:")
    print(f"  均值: {mean} (非零，是标准高斯分布的平移)")
    print(f"  协方差矩阵: 单位矩阵 (2x2)")
    print(f"  特征值: {np.linalg.eigvals(cov)} (应为[1, 1])")

# 10. 绘制简洁版本
plt.figure(figsize=(10, 8))
# 绘制散点图
plt.scatter(samples[:, 0], samples[:, 1], c='red', s=15, alpha=0.7, 
           label='采样点', edgecolor='black', linewidth=0.5)
# 绘制概率密度等高线
contour = plt.contour(X, Y, Z, levels=15, colors='black', linewidths=1.5, alpha=0.9)
plt.clabel(contour, inline=True, fontsize=9, colors='black')
plt.xlabel('X', fontsize=14)
plt.ylabel('Y', fontsize=14)
plt.title('三个标准高斯分布（不同均值）加权求和', fontsize=16, pad=20)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.tight_layout()

# 添加颜色条表示概率密度
sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=Z.min(), vmax=Z.max()))
sm.set_array([])
cbar = plt.colorbar(sm, ax=plt.gca(), shrink=0.8)
cbar.set_label('概率密度', fontsize=12)

# 保存简洁版本
plt.savefig('standard_gaussian_mixture_simple.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# 11. 再绘制一个更干净的版本
plt.figure(figsize=(9, 7))
# 绘制散点图
plt.scatter(samples[:, 0], samples[:, 1], c='red', s=20, alpha=0.8, 
           label='采样点', edgecolor='black', linewidth=0.8)
# 绘制概率密度等高线
contour = plt.contour(X, Y, Z, levels=12, colors='black', linewidths=2, alpha=0.9)
plt.clabel(contour, inline=True, fontsize=10, colors='black')
plt.xlabel('X', fontsize=14)
plt.ylabel('Y', fontsize=14)
plt.title('标准高斯混合分布采样与概率密度等高线', fontsize=16, pad=20)
plt.legend(fontsize=12, loc='upper right')
plt.grid(True, alpha=0.2)
plt.axis('equal')
plt.tight_layout()

# 保存最终版本
plt.savefig('final_standard_gaussian_mixture_contour_scatter.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# 12. 绘制真正的标准高斯分布（均值为0，方差为I）作为对比
print("\n=== 对比：真正的标准高斯分布（均值0，方差I）===")
# 从标准高斯分布采样
standard_samples = np.random.multivariate_normal([0, 0], np.eye(2), num_samples)

# 计算标准高斯分布的概率密度
standard_Z = multivariate_normal(mean=[0, 0], cov=np.eye(2)).pdf(grid_points).reshape(X.shape)

# 绘制对比图
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 左侧：真正的标准高斯分布
ax1 = axes[0]
ax1.scatter(standard_samples[:, 0], standard_samples[:, 1], c='blue', s=10, alpha=0.5, 
           label='待检测环境声音样本特征h映射值', edgecolor='black', linewidth=0.2)
contour_std = ax1.contour(X, Y, standard_Z, levels=10, colors='black', linewidths=1.5, alpha=0.8)
ax1.clabel(contour_std, inline=True, fontsize=8, colors='black')
ax1.set_xlabel('X', fontsize=12)
ax1.set_ylabel('Y', fontsize=12)
ax1.set_title('环境声音样本原型p_k映射值相似度等高线图', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.axis('equal')

# 右侧：平移后的标准高斯分布混合
ax2 = axes[1]
ax2.scatter(samples[:, 0], samples[:, 1], c='red', s=10, alpha=0.5, 
           label='混合高斯采样点', edgecolor='black', linewidth=0.2)
contour_mix = ax2.contour(X, Y, Z, levels=10, colors='black', linewidths=1.5, alpha=0.8)
ax2.clabel(contour_mix, inline=True, fontsize=8, colors='black')
ax2.set_xlabel('X', fontsize=12)
ax2.set_ylabel('Y', fontsize=12)
ax2.set_title('平移后的标准高斯分布混合\n(不同均值，相同方差=I)', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.axis('equal')

plt.suptitle('标准高斯分布对比：单一分布 vs 混合分布', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('gaussian_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()