import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
fm.fontManager = fm.FontManager()
from scipy.stats import multivariate_normal
import matplotlib.cm as cm
import os
# 设置中文字体支持
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
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

# 4. 按照概率分布生成采样点
num_samples = 1000

# 方法1: 通过逆变换采样
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

# 方法2: 接受-拒绝采样 (备选，确保采样准确)
# 但上面的逆变换采样更高效且准确

# 5. 创建三维图形
fig = plt.figure(figsize=(15, 10))

# 5.1 第一个子图：三维曲面图
ax1 = fig.add_subplot(221, projection='3d')
surf1 = ax1.plot_surface(X, Y, Z, cmap='viridis', 
                        alpha=0.8, linewidth=0, antialiased=True)
ax1.set_xlabel('X', fontsize=12)
ax1.set_ylabel('Y', fontsize=12)
ax1.set_zlabel('概率密度 p_k(h)', fontsize=12)
ax1.set_title('标准化流混合模型的概率密度曲面', fontsize=14, pad=20)
fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10, label='概率密度')

# 5.2 第二个子图：曲面图 + 采样点
ax2 = fig.add_subplot(222, projection='3d')
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
ax3 = fig.add_subplot(223)
contour = ax3.contourf(X, Y, Z, levels=50, cmap='viridis')
ax3.scatter(samples[:, 0], samples[:, 1], c='red', s=5, alpha=0.5, label='采样点')
ax3.set_xlabel('X', fontsize=12)
ax3.set_ylabel('Y', fontsize=12)
ax3.set_title('概率密度俯视图 (热力图)', fontsize=14)
ax3.legend()
fig.colorbar(contour, ax=ax3, label='概率密度')

# 5.4 第四个子图：概率密度等高线
ax4 = fig.add_subplot(224)
contour_lines = ax4.contour(X, Y, Z, levels=15, colors='black', alpha=0.5)
ax4.clabel(contour_lines, inline=True, fontsize=8)
ax4.scatter(samples[:, 0], samples[:, 1], c='red', s=5, alpha=0.6, label='采样点')
ax4.set_xlabel('X', fontsize=12)
ax4.set_ylabel('Y', fontsize=12)
ax4.set_title('概率密度等高线图', fontsize=14)
ax4.legend()

plt.suptitle('基于标准化流混合模型的概率密度可视化与采样', fontsize=16, y=1.02)
plt.tight_layout()

# 6. 保存图像
output_dir = 'flow_model_visualization'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'normalizing_flow_density.png')
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
from mpl_toolkits.mplot3d import Axes3D

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

# 10. 验证采样分布的正确性
print("\n=== 采样分布验证 ===")
print(f"采样点数: {len(samples)}")
print(f"采样点范围: X∈[{samples[:, 0].min():.2f}, {samples[:, 0].max():.2f}], "
      f"Y∈[{samples[:, 1].min():.2f}, {samples[:, 1].max():.2f}]")

# 计算经验分布与理论分布的KL散度（近似验证）
from scipy.special import kl_div
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