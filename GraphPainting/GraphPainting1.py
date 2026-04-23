import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
fm.fontManager = fm.FontManager()
from scipy.stats import multivariate_normal
import matplotlib.cm as cm
import os
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import kl_div

# 设置中文字体支持
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子以便结果可重现
np.random.seed(42)

# 1. 定义标准化流模型相关的参数
class AffineFlow:
    """简单的仿射流变换"""
    def __init__(self, input_dim=2):
        # 生成一个可逆的线性变换矩阵A
        # 先随机生成正交矩阵确保可逆
        U, _ = np.linalg.qr(np.random.randn(input_dim, input_dim))
        S = np.diag(np.exp(np.random.randn(input_dim) * 0.5 + 0.5))  # 正定对角矩阵
        self.A = U @ S
        self.b = np.random.randn(input_dim) * 2
        
    def forward(self, h):
        """正向变换: z = A * h + b"""
        z = h @ self.A.T + self.b
        log_det = np.log(np.abs(np.linalg.det(self.A)))
        return z, log_det
    
    def inverse(self, z):
        """逆向变换: h = A^{-1} * (z - b)"""
        A_inv = np.linalg.inv(self.A)
        h = (z - self.b) @ A_inv.T
        return h

def q_km(h, flow, standard_normal):
    """
    计算标准化流模型的概率密度 q_km(h)
    q_km(h) = N(f_km(h); 0, I) * |det(J_{f_km}(h))|
    """
    # 正向变换
    z, log_det = flow.forward(h)
    
    # 计算标准正态分布下的概率密度
    # 注意：multivariate_normal的pdf是直接计算概率密度，不是对数概率密度
    # 我们使用对数计算以避免数值下溢
    log_prob_z = standard_normal.logpdf(z)
    
    # 概率密度 = exp(log_prob_z) * exp(log_det) = exp(log_prob_z + log_det)
    log_prob_h = log_prob_z + log_det
    return np.exp(log_prob_h)

def p_k(h, flows, pi, standard_normal):
    """
    计算混合概率密度 p_k(h) = ∑ π_km * q_km(h)
    """
    total_density = np.zeros(h.shape[0])
    
    for m, flow in enumerate(flows):
        density_km = q_km(h, flow, standard_normal)
        total_density += pi[m] * density_km
        
    return total_density

def pf_qf_k(h, flows, pi, standard_normal):
    """
    计算pf_k(h)和qf_km(h)的映射值
    
    参数:
    h: 输入点，形状为 (n_points, 2)
    flows: 仿射流模型列表
    pi: 混合系数
    standard_normal: 标准正态分布对象
    
    返回:
    pf_value: pf_k(h)的映射值，形状为 (n_points, 2)
    qf_values: 每个混合成分的qf_km(h)映射值，形状为 (n_components, n_points, 2)
    """
    n_points = h.shape[0]
    n_components = len(flows)
    
    # 初始化数组
    pf_value = np.zeros((n_points, 2))
    qf_values = np.zeros((n_components, n_points, 2))
    
    for m, flow in enumerate(flows):
        # 计算qf_km(h) = f_km(h) * |det(J_{f_km}(h))|
        z, log_det = flow.forward(h)
        det_value = np.exp(log_det)
        
        # qf_km(h) = f_km(h) * |det(J_{f_km}(h))| = z * exp(log_det)
        qf_values[m] = z * det_value  # 广播det_value到每个维度
        
        # 累积到pf_k(h)
        pf_value += pi[m] * qf_values[m]
    
    return pf_value, qf_values

# 2. 随机生成参数
num_components = 3  # 混合成分数 M_k
input_dim = 2  # 输入维度

# 生成混合系数 π_km (归一化使得和为1)
pi = np.random.dirichlet(np.ones(num_components) * 1.5)

# 为每个混合成分创建一个仿射流模型
flows = [AffineFlow(input_dim) for _ in range(num_components)]

# 标准正态分布对象
standard_normal = multivariate_normal(mean=np.zeros(input_dim), cov=np.eye(input_dim))

# 3. 创建网格用于计算概率密度
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)

# 将网格点展平以便批量计算
grid_points = np.column_stack([X.ravel(), Y.ravel()])

# 计算每个网格点的概率密度
Z = p_k(grid_points, flows, pi, standard_normal)
Z = Z.reshape(X.shape)

# 计算pf_k(h)和qf_km(h)的映射值
pf_values, qf_values = pf_qf_k(grid_points, flows, pi, standard_normal)

# 计算pf_k(h)的模长（作为标量表示）
pf_magnitude = np.linalg.norm(pf_values, axis=1).reshape(X.shape)

# 计算每个混合成分的qf_km(h)的模长
qf_magnitudes = np.zeros((num_components, X.shape[0], X.shape[1]))
for m in range(num_components):
    qf_magnitudes[m] = np.linalg.norm(qf_values[m], axis=1).reshape(X.shape)

# 4. 按照概率分布生成采样点
num_samples = 1000

# 通过逆变换采样
samples = []
for _ in range(num_samples):
    # 1) 根据混合系数选择一个成分
    m = np.random.choice(num_components, p=pi)
    
    # 2) 从标准正态分布采样 z
    z_sample = np.random.multivariate_normal(np.zeros(input_dim), np.eye(input_dim))
    
    # 3) 通过逆变换得到 h = f^{-1}(z)
    h_sample = flows[m].inverse(z_sample.reshape(1, -1)).flatten()
    samples.append(h_sample)

samples = np.array(samples)

# 5. 创建三维图形
fig = plt.figure(figsize=(20, 12))

# 5.1 第一个子图：三维曲面图
ax1 = fig.add_subplot(231, projection='3d')
surf1 = ax1.plot_surface(X, Y, Z, cmap='viridis', 
                        alpha=0.8, linewidth=0, antialiased=True)
ax1.set_xlabel('X', fontsize=12)
ax1.set_ylabel('Y', fontsize=12)
ax1.set_zlabel('概率密度 p_k(h)', fontsize=12)
ax1.set_title('标准化流混合模型的概率密度曲面', fontsize=14, pad=20)
fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10, label='概率密度')

# 5.2 第二个子图：曲面图 + 采样点
ax2 = fig.add_subplot(232, projection='3d')
surf2 = ax2.plot_surface(X, Y, Z, cmap='viridis', 
                        alpha=0.6, linewidth=0, antialiased=True)
# 绘制采样点
ax2.scatter(samples[:, 0], samples[:, 1], np.zeros(num_samples), 
           c='red', s=5, alpha=0.6, label='采样点')
ax2.set_xlabel('X', fontsize=12)
ax2.set_ylabel('Y', fontsize=12)
ax2.set_zlabel('概率密度 p_k(h)', fontsize=12)
ax2.set_title('曲面图 + 采样点 (X-Y平面)', fontsize=14, pad=20)
ax2.legend()

# 5.3 第三个子图：俯视图 (热力图)
ax3 = fig.add_subplot(233)
contour = ax3.contourf(X, Y, Z, levels=50, cmap='viridis')
ax3.scatter(samples[:, 0], samples[:, 1], c='red', s=5, alpha=0.5, label='采样点')
ax3.set_xlabel('X', fontsize=12)
ax3.set_ylabel('Y', fontsize=12)
ax3.set_title('概率密度俯视图 (热力图)', fontsize=14)
ax3.legend()
fig.colorbar(contour, ax=ax3, label='概率密度')

# 5.4 第四个子图：概率密度等高线
ax4 = fig.add_subplot(234)
contour_lines = ax4.contour(X, Y, Z, levels=15, colors='black', alpha=0.5)
ax4.clabel(contour_lines, inline=True, fontsize=8)
ax4.scatter(samples[:, 0], samples[:, 1], c='red', s=5, alpha=0.6, label='采样点')
ax4.set_xlabel('X', fontsize=12)
ax4.set_ylabel('Y', fontsize=12)
ax4.set_title('概率密度等高线图', fontsize=14)
ax4.legend()

# 5.5 第五个子图：pf_k(h)映射值与概率密度p_k(h)关系的等高线图
ax5 = fig.add_subplot(235)
# 用颜色表示pf_k(h)的模长
# 叠加概率密度等高线
density_contour = ax5.contour(X, Y, Z, levels=10, colors='black', linewidths=1.5, alpha=0.8)
ax5.clabel(density_contour, inline=True, fontsize=8, colors='black')
ax5.scatter(samples[:, 0], samples[:, 1], c='red', s=5, alpha=0.6, label='采样点')
ax5.set_xlabel('X', fontsize=12)
ax5.set_ylabel('Y', fontsize=12)
ax5.set_title('pf_k(h)映射值与概率密度关系', fontsize=14)
ax5.legend()

# 5.6 第六个子图：qf_km(h)映射值对比
ax6 = fig.add_subplot(236)
# 计算混合成分的加权平均qf映射值
weighted_qf = np.zeros_like(pf_magnitude)
for m in range(num_components):
    weighted_qf += pi[m] * qf_magnitudes[m]

# 绘制加权平均qf映射值
# 叠加概率密度等高线
density_contour2 = ax6.contour(X, Y, Z, levels=10, colors='black', linewidths=1.5, alpha=0.8)
ax6.clabel(density_contour2, inline=True, fontsize=8, colors='black')
ax6.scatter(samples[:, 0], samples[:, 1], c='green', s=5, alpha=0.6, label='采样点')
ax6.set_xlabel('X', fontsize=12)
ax6.set_ylabel('Y', fontsize=12)
ax6.set_title('加权平均qf_km(h)映射值与概率密度', fontsize=14)
ax6.legend()

plt.suptitle('基于标准化流混合模型的概率密度可视化与映射值分析', fontsize=16, y=1.02)
plt.tight_layout()

# 6. 保存图像
output_dir = 'flow_model_visualization'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'normalizing_flow_density_with_mapping.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"图像已保存至: {output_path}")

# 7. 显示图像
plt.show()

# 8. 打印生成参数
print("\n=== 生成的标准化流模型参数 ===")
print(f"混合成分数 M_k: {num_components}")
print(f"混合系数 π_km: {pi}")

for m, flow in enumerate(flows):
    print(f"\n--- 成分 {m+1} ---")
    print(f"变换矩阵 A_{m+1}:")
    print(flow.A)
    print(f"偏置向量 b_{m+1}: {flow.b}")
    print(f"行列式 |det(A_{m+1})|: {np.abs(np.linalg.det(flow.A)):.4f}")

# 9. 额外：生成一个单独的3D交互图（可选）
fig2 = plt.figure(figsize=(12, 8))
ax = fig2.add_subplot(111, projection='3d')

# 使用渐变色
colors = Z / Z.max()
norm = plt.Normalize(colors.min(), colors.max())
cmap = cm.viridis

# 绘制曲面
surf = ax.plot_surface(X, Y, Z, facecolors=cmap(norm(colors)),
                      alpha=0.8, linewidth=0.5, antialiased=True)

# 绘制采样点
ax.scatter(samples[:, 0], samples[:, 1], np.zeros(num_samples), 
          c='red', s=20, alpha=0.8, depthshade=True, label='采样点 (X-Y平面)')

# 添加从采样点到曲面的投影线
for i in range(0, len(samples), 20):  # 每20个点画一个投影线
    sample = samples[i]
    density = p_k(sample.reshape(1, -1), flows, pi, standard_normal)[0]
    ax.plot([sample[0], sample[0]], [sample[1], sample[1]], 
            [0, density], 'gray', alpha=0.3, linewidth=0.5)

ax.set_xlabel('X', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.set_zlabel('概率密度 p_k(h)', fontsize=12)
ax.set_title('标准化流混合模型概率密度曲面与采样点', fontsize=14, pad=20)
ax.legend()

# 添加颜色条
mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
mappable.set_array(Z)
fig2.colorbar(mappable, ax=ax, shrink=0.5, aspect=10, label='概率密度')

# 设置视角
ax.view_init(elev=30, azim=45)

plt.tight_layout()
interactive_path = os.path.join(output_dir, 'flow_density_interactive.png')
plt.savefig(interactive_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n交互式视图已保存至: {interactive_path}")

plt.show()

# 10. 创建专门的映射值分析图
fig3, axes = plt.subplots(2, 3, figsize=(18, 10))

# 10.1 概率密度等高线
ax1 = axes[0, 0]
contour1 = ax1.contourf(X, Y, Z, levels=50, cmap='viridis')
ax1.set_xlabel('X', fontsize=12)
ax1.set_ylabel('Y', fontsize=12)
ax1.set_title('概率密度 p_k(h)', fontsize=14)
fig3.colorbar(contour1, ax=ax1, label='概率密度')

# 10.2 pf_k(h)映射值的x分量
ax2 = axes[0, 1]
pf_x = pf_values[:, 0].reshape(X.shape)
contour2 = ax2.contourf(X, Y, pf_x, levels=50, cmap='coolwarm')
ax2.set_xlabel('X', fontsize=12)
ax2.set_ylabel('Y', fontsize=12)
ax2.set_title('pf_k(h) 的 x 分量', fontsize=14)
fig3.colorbar(contour2, ax=ax2, label='pf_k(h) x 分量')

# 10.3 pf_k(h)映射值的y分量
ax3 = axes[0, 2]
pf_y = pf_values[:, 1].reshape(X.shape)
contour3 = ax3.contourf(X, Y, pf_y, levels=50, cmap='coolwarm')
ax3.set_xlabel('X', fontsize=12)
ax3.set_ylabel('Y', fontsize=12)
ax3.set_title('pf_k(h) 的 y 分量', fontsize=14)
fig3.colorbar(contour3, ax=ax3, label='pf_k(h) y 分量')

# 10.4 pf_k(h)映射值模长
ax4 = axes[1, 0]
contour4 = ax4.contourf(X, Y, pf_magnitude, levels=50, cmap='plasma')
# 叠加概率密度等高线
density_lines = ax4.contour(X, Y, Z, levels=10, colors='white', linewidths=1, alpha=0.7)
ax4.clabel(density_lines, inline=True, fontsize=8, colors='white')
ax4.set_xlabel('X', fontsize=12)
ax4.set_ylabel('Y', fontsize=12)
ax4.set_title('pf_k(h) 映射值模长与概率密度等高线', fontsize=14)
fig3.colorbar(contour4, ax=ax4, label='pf_k(h) 模长')

# 10.5 概率密度与pf_k(h)模长的关系散点图
ax5 = axes[1, 1]
scatter = ax5.scatter(grid_points[:, 0], grid_points[:, 1], 
                     c=pf_magnitude.flatten(), s=5, cmap='plasma', 
                     alpha=0.7, edgecolor='none')
# 添加采样点
ax5.scatter(samples[:, 0], samples[:, 1], c='red', s=20, alpha=0.6, 
           edgecolor='black', linewidth=0.5, label='采样点')
ax5.set_xlabel('X', fontsize=12)
ax5.set_ylabel('Y', fontsize=12)
ax5.set_title('pf_k(h) 映射值模长分布与采样点', fontsize=14)
ax5.legend()
# fig3.colorbar(scatter, ax=ax5, label='pf_k(h) 模长')

# 10.6 概率密度与映射值的关系（3D散点图）
ax6 = axes[1, 2]
# 从网格点中随机选择一部分点进行可视化
np.random.seed(42)
sample_indices = np.random.choice(len(grid_points), 500, replace=False)
sample_points = grid_points[sample_indices]
sample_density = Z.flatten()[sample_indices]
sample_pf_mag = pf_magnitude.flatten()[sample_indices]

# 创建3D散点图
scatter = ax6.scatter(sample_points[:, 0], sample_points[:, 1], c='white',
                       edgecolor='black', linewidth=0.3)
ax6.set_xlabel('X', fontsize=12)
ax6.set_ylabel('Y', fontsize=12)
ax6.set_ylabel('pf_k(h) 模', fontsize=12)
ax6.set_title('概率密度与映射值模长的3D关系', fontsize=14)

plt.suptitle('标准化流混合模型：映射值分析', fontsize=16, y=1.02)
plt.tight_layout()

# 保存映射值分析图
mapping_analysis_path = os.path.join(output_dir, 'mapping_value_analysis.png')
plt.savefig(mapping_analysis_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"映射值分析图已保存至: {mapping_analysis_path}")

plt.show()

# 11. 验证采样分布的正确性
print("\n=== 采样分布验证 ===")
print(f"采样点数: {len(samples)}")
print(f"采样点范围: X∈[{samples[:, 0].min():.2f}, {samples[:, 0].max():.2f}], "
      f"Y∈[{samples[:, 1].min():.2f}, {samples[:, 1].max():.2f}]")

# 计算经验分布与理论分布的KL散度（近似验证）
hist, x_edges, y_edges = np.histogram2d(samples[:, 0], samples[:, 1], 
                                       bins=20, range=[[-5, 5], [-5, 5]], density=True)

# 在网格中点计算理论概率密度
x_centers = (x_edges[:-1] + x_edges[1:]) / 2
y_centers = (y_edges[:-1] + y_edges[1:]) / 2
Xc, Yc = np.meshgrid(x_centers, y_centers)
grid_centers = np.column_stack([Xc.ravel(), Yc.ravel()])
theoretical_density = p_k(grid_centers, flows, pi, standard_normal)
theoretical_density = theoretical_density.reshape(Xc.shape)

# 归一化理论密度（使其在直方图范围内积分为1）
theoretical_hist = theoretical_density / theoretical_density.sum()

# 比较直方图
empirical_hist = hist / hist.sum()

# 避免零值
epsilon = 1e-10
empirical_hist = empirical_hist + epsilon
empirical_hist = empirical_hist / empirical_hist.sum()
theoretical_hist = theoretical_hist + epsilon
theoretical_hist = theoretical_hist / theoretical_hist.sum()

# 计算KL散度
kl_divergence = np.sum(kl_div(empirical_hist.flatten(), theoretical_hist.flatten()))
print(f"经验分布与理论分布的KL散度: {kl_divergence:.4f}")
print("(KL散度越小，说明采样分布与理论分布越接近)")

# 12. 输出映射值的统计信息
print("\n=== 映射值统计信息 ===")
print(f"pf_k(h)映射值范围:")
print(f"  X分量: [{pf_values[:, 0].min():.4f}, {pf_values[:, 0].max():.4f}]")
print(f"  Y分量: [{pf_values[:, 1].min():.4f}, {pf_values[:, 1].max():.4f}]")
print(f"  模长: [{pf_magnitude.min():.4f}, {pf_magnitude.max():.4f}]")

for m in range(num_components):
    qf_mag = qf_magnitudes[m]
    print(f"\n成分 {m+1} 的qf_km(h)映射值模长:")
    print(f"  范围: [{qf_mag.min():.4f}, {qf_mag.max():.4f}]")
    print(f"  均值: {qf_mag.mean():.4f}, 标准差: {qf_mag.std():.4f}")