"""
题目四：编写 SMO 算法实现线性 SVM 分类器，对 iris 数据集进行二分类

具体内容：
(1) 选取两个特征和两类数据进行二分类
(2) 划分数据（分成训练集和测试集）
(3) 数据归一化
(4) 训练模型（参考程序模板）
(5) 输出：SVM 对偶问题目标函数的最小值α，决策函数的参数和截距，支持向量等
(6) 可视化：散点图可视化训练数据样本，画出决策面和 2 个最大间隔边界，标出支持向量
(7) 测试集数据进行预测，评估模型性能
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import sys

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("题目四：SMO 算法实现线性 SVM 分类器 - iris 二分类")
print("=" * 80)

# ============================================================================
# 定义简化版 SMO 算法的 SVM 类11
# ============================================================================

class LinearSVMSMO:
    """使用 SMO 算法实现的线性 SVM 分类器"""
    
    def __init__(self, C=1.0, max_iter=200, tol=1e-3, kernel_type='linear'):
        self.C = C  # 惩罚系数
        self.max_iter = max_iter  # 最大迭代次数
        self.tol = tol  # 容差
        self.kernel_type = kernel_type
        
        # 模型参数
        self.alpha = None  # 拉格朗日乘数
        self.b = 0  # 截距
        self.w = None  # 权重向量
        self.support_vectors_idx = None  # 支持向量索引
        self.support_vectors = None  # 支持向量坐标
        self.X_train = None
        self.y_train = None
        
    def _kernel(self, x1, x2):
        """线性核函数"""
        if self.kernel_type == 'linear':
            return np.dot(x1, x2.T)
        raise ValueError(f"Unsupported kernel: {self.kernel_type}")
    
    def fit(self, X, y):
        """
        使用 SMO 算法训练 SVM
        
        参数：
        - X: 训练数据，形状 (n_samples, n_features)
        - y: 标签，值为 +1 或 -1
        """
        m, n = X.shape
        self.X_train = X
        self.y_train = y
        
        # 初始化 alpha
        self.alpha = np.zeros(m)
        self.b = 0
        
        # 计算核矩阵 K
        K = self._kernel(X, X)
        
        # 计算 E 缓存
        E = np.zeros(m)
        
        # SMO 主循环
        iteration = 0
        for iteration in range(self.max_iter):
            num_changed_alphas = 0
            
            for i in range(m):
                # 计算 E_i（预测误差）
                E[i] = np.dot(self.alpha * y, K[i, :]) + self.b - y[i]
                
                # 检查 KKT 条件
                r_i = y[i] * E[i]
                
                if ((r_i < -self.tol and self.alpha[i] < self.C) or
                    (r_i > self.tol and self.alpha[i] > 0)):
                    
                    # 选择第二个变量 j（简单策略：选择最大 |E_i - E_j| 的样本）
                    j = self._select_second_alpha(i, E)
                    
                    if j == i:
                        continue
                    
                    # 计算 E_j
                    E[j] = np.dot(self.alpha * y, K[j, :]) + self.b - y[j]
                    
                    # 保存旧 alpha 值
                    alpha_i_old = self.alpha[i].copy()
                    alpha_j_old = self.alpha[j].copy()
                    
                    # 计算 alpha_j 的上下界
                    if y[i] != y[j]:
                        L = max(0, self.alpha[j] - self.alpha[i])
                        H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
                    else:
                        L = max(0, self.alpha[i] + self.alpha[j] - self.C)
                        H = min(self.C, self.alpha[i] + self.alpha[j])
                    
                    if L == H:
                        continue
                    
                    # 计算 eta（二阶导数）
                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    
                    if eta >= 0:
                        continue
                    
                    # 更新 alpha_j
                    self.alpha[j] -= y[j] * (E[i] - E[j]) / eta
                    self.alpha[j] = np.clip(self.alpha[j], L, H)
                    
                    # 检查是否有实质性改变
                    if abs(self.alpha[j] - alpha_j_old) < 1e-7:
                        continue
                    
                    # 更新 alpha_i
                    self.alpha[i] += y[i] * y[j] * (alpha_j_old - self.alpha[j])
                    
                    # 更新 b
                    b1 = self.b - E[i] - y[i] * (self.alpha[i] - alpha_i_old) * K[i, i] - \
                         y[j] * (self.alpha[j] - alpha_j_old) * K[i, j]
                    b2 = self.b - E[j] - y[i] * (self.alpha[i] - alpha_i_old) * K[i, j] - \
                         y[j] * (self.alpha[j] - alpha_j_old) * K[j, j]
                    
                    if 0 < self.alpha[i] < self.C:
                        self.b = b1
                    elif 0 < self.alpha[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2
                    
                    num_changed_alphas += 1
            
            if num_changed_alphas == 0:
                print(f"SMO 算法在第 {iteration + 1} 次迭代时收敛")
                break
        
        # 计算权重向量 w
        self.w = np.dot(self.alpha * y, X)
        
        # 提取支持向量（alpha > 0 的样本）
        self.support_vectors_idx = np.where(self.alpha > 1e-5)[0]
        self.support_vectors = X[self.support_vectors_idx]
        
        return self
    
    def _select_second_alpha(self, i, E):
        """选择第二个 alpha（选择使得 |E_i - E_j| 最大的 j）"""
        valid_indices = np.where((self.alpha != 0) | (self.alpha != self.C))[0]
        if len(valid_indices) < 2:
            j = np.random.choice(np.arange(len(E)))
        else:
            max_delta = -np.inf
            j = i
            for jj in valid_indices:
                if jj != i:
                    delta = abs(E[i] - E[jj])
                    if delta > max_delta:
                        max_delta = delta
                        j = jj
        return j
    
    def decision_function(self, X):
        """计算决策函数 f(x) = w^T @ x + b"""
        return np.dot(X, self.w) + self.b
    
    def predict(self, X):
        """预测类别（+1 或 -1）"""
        decision = self.decision_function(X)
        return np.sign(decision)
    
    def score(self, X, y):
        """计算准确率"""
        predictions = self.predict(X)
        return np.mean(predictions == y)


# ============================================================================
# (1) 选取两个特征和两类数据
# ============================================================================
print("\n【第一步：数据准备】\n")

iris = load_iris()
X = iris.data[:, 2:4]  # 特征：花瓣长度和花瓣宽度
y = iris.target

# 二分类：选择类别 1 和 2
mask = (y == 1) | (y == 2)
X = X[mask]
y = y[mask]

# 将标签转换为 +1 和 -1
y = np.where(y == 1, -1, 1)

print(f"特征：花瓣长度 (petal length) 和花瓣宽度 (petal width)")
print(f"类别：Versicolor (-1, 50 个) 和 Virginica (+1, 50 个)")
print(f"总样本数：{X.shape[0]}")
print(f"特征维度：{X.shape[1]}")

# ============================================================================
# (2) 划分数据
# ============================================================================
print("\n【第二步：数据划分】\n")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"训练集大小：{X_train.shape[0]}")
print(f"测试集大小：{X_test.shape[0]}")

# ============================================================================
# (3) 数据归一化
# ============================================================================
print("\n【第三步：数据归一化】\n")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("特征已进行标准化（StandardScaler）")
print(f"训练集 - 均值: {X_train_scaled.mean(axis=0)}, 方差: {X_train_scaled.std(axis=0)}")

# ============================================================================
# (4) 训练模型
# ============================================================================
print("\n【第四步：训练 SVM 模型（SMO 算法）】\n")

print("开始训练 SVM 模型...")
svm_model = LinearSVMSMO(C=1.0, max_iter=200, tol=1e-3)
svm_model.fit(X_train_scaled, y_train)
print("模型训练完成！")

# ============================================================================
# (5) 输出模型参数
# ============================================================================
print("\n【第五步：模型参数输出】\n")

print("=" * 80)
print("SVM 对偶问题求解结果：")
print("=" * 80)

print(f"\n1. 拉格朗日乘数 α:")
print("-" * 80)
print(f"   非零 α 的个数（支持向量数）: {len(svm_model.support_vectors_idx)}")
print(f"   所有 α 的和: {np.sum(svm_model.alpha):.6f}")
print(f"   α 的统计信息:")
print(f"     - 最小值: {np.min(svm_model.alpha):.6f}")
print(f"     - 最大值: {np.max(svm_model.alpha):.6f}")
print(f"     - 平均值: {np.mean(svm_model.alpha):.6f}")

# 显示前 10 个非零 alpha
nonzero_alpha = svm_model.alpha[svm_model.alpha > 1e-5]
print(f"\n   非零 α 值（前 10 个）:")
for idx, alpha_val in enumerate(sorted(nonzero_alpha, reverse=True)[:10]):
    print(f"     α[{idx+1}] = {alpha_val:.6f}")

print(f"\n2. 决策函数参数:")
print("-" * 80)
print(f"   权重向量 w: {svm_model.w}")
print(f"   ||w||: {np.linalg.norm(svm_model.w):.6f}")
print(f"   截距 b: {svm_model.b:.6f}")
print(f"   间隔宽度 (2/||w||): {2.0 / np.linalg.norm(svm_model.w):.6f}")

print(f"\n3. 支持向量:")
print("-" * 80)
print(f"   支持向量个数: {len(svm_model.support_vectors_idx)}")
print(f"   支持向量在训练集中的索引: {svm_model.support_vectors_idx[:20]}")
if len(svm_model.support_vectors_idx) > 20:
    print(f"   ... (共 {len(svm_model.support_vectors_idx)} 个)")
print(f"\n   支持向量坐标（前 5 个）:")
for i, sv_idx in enumerate(svm_model.support_vectors_idx[:5]):
    print(f"     支持向量 {i+1}: {X_train_scaled[sv_idx]} (标签: {y_train[sv_idx]})")

# ============================================================================
# (6) 可视化
# ============================================================================
print("\n【第六步：可视化】\n")

fig, ax = plt.subplots(figsize=(12, 10))

# 创建网格
x1_min, x1_max = X_train_scaled[:, 0].min() - 0.5, X_train_scaled[:, 0].max() + 0.5
x2_min, x2_max = X_train_scaled[:, 1].min() - 0.5, X_train_scaled[:, 1].max() + 0.5

xx, yy = np.meshgrid(np.linspace(x1_min, x1_max, 300),
                     np.linspace(x2_min, x2_max, 300))

# 计算网格点的决策函数值
Z = svm_model.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 绘制决策边界
ax.contour(xx, yy, Z, levels=[0], linewidths=2.5, colors='black', label='决策边界')

# 绘制间隔边界
ax.contour(xx, yy, Z, levels=[-1, 1], linewidths=2, colors='black', 
           linestyles='dashed', alpha=0.7, label='最大间隔边界')

# 绘制填充区域
ax.contourf(xx, yy, Z, levels=np.linspace(Z.min(), Z.max(), 20), 
            cmap='RdBu_r', alpha=0.3)

# 绘制训练数据
scatter1 = ax.scatter(X_train_scaled[y_train == -1, 0], 
                     X_train_scaled[y_train == -1, 1],
                     c='blue', marker='o', s=80, edgecolors='k', 
                     alpha=0.7, label='类别 1 (Versicolor)')
scatter2 = ax.scatter(X_train_scaled[y_train == 1, 0], 
                     X_train_scaled[y_train == 1, 1],
                     c='green', marker='o', s=80, edgecolors='k', 
                     alpha=0.7, label='类别 2 (Virginica)')

# 标出支持向量
sv_scatter = ax.scatter(X_train_scaled[svm_model.support_vectors_idx, 0],
                       X_train_scaled[svm_model.support_vectors_idx, 1],
                       s=300, linewidth=2, facecolors='none', 
                       edgecolors='red', label='支持向量')

ax.set_xlabel('花瓣长度 (标准化)', fontsize=12, fontweight='bold')
ax.set_ylabel('花瓣宽度 (标准化)', fontsize=12, fontweight='bold')
ax.set_title('SMO 算法实现的 SVM 线性分类器 - Iris 二分类', 
             fontsize=14, fontweight='bold')
ax.legend(loc='upper left', fontsize=11)
ax.grid(True, alpha=0.3)

# 添加文本信息
textstr = f'模型参数\nw = [{svm_model.w[0]:.4f}, {svm_model.w[1]:.4f}]\nb = {svm_model.b:.4f}\n支持向量数: {len(svm_model.support_vectors_idx)}\n间隔宽度: {2.0/np.linalg.norm(svm_model.w):.4f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props, family='monospace')

plt.tight_layout()
plt.savefig('svm_smo_iris_classification.png', dpi=300, bbox_inches='tight')
print("可视化图像已保存为 'svm_smo_iris_classification.png'")

# ============================================================================
# (7) 测试集预测和性能评估
# ============================================================================
print("\n【第七步：测试集预测和性能评估】\n")

print("=" * 80)
print("模型性能评估：")
print("=" * 80)

# 训练集准确率
y_train_pred = svm_model.predict(X_train_scaled)
train_accuracy = svm_model.score(X_train_scaled, y_train)

# 测试集准确率
y_test_pred = svm_model.predict(X_test_scaled)
test_accuracy = svm_model.score(X_test_scaled, y_test)

print(f"\n1. 准确率:")
print("-" * 80)
print(f"   训练集准确率: {train_accuracy:.4f} ({int(train_accuracy * len(y_train))}/{len(y_train)})")
print(f"   测试集准确率: {test_accuracy:.4f} ({int(test_accuracy * len(y_test))}/{len(y_test)})")

print(f"\n2. 分类报告（测试集）:")
print("-" * 80)
print(classification_report(y_test, y_test_pred, 
                          target_names=['Versicolor', 'Virginica'],
                          digits=4))

print(f"\n3. 混淆矩阵（测试集）:")
print("-" * 80)
cm = confusion_matrix(y_test, y_test_pred)
print(cm)

# 绘制混淆矩阵
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

ax.set_xlabel('预测标签', fontsize=12, fontweight='bold')
ax.set_ylabel('真实标签', fontsize=12, fontweight='bold')
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Versicolor', 'Virginica'])
ax.set_yticklabels(['Versicolor', 'Virginica'])

# 添加数值
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=14, fontweight='bold')

ax.set_title(f'混淆矩阵 - 测试集准确率: {test_accuracy:.4f}', fontsize=13, fontweight='bold')
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig('svm_smo_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("\n混淆矩阵图像已保存为 'svm_smo_confusion_matrix.png'")

print("\n" + "=" * 80)
print("【总结】")
print("=" * 80)
print(f"""
本实验使用 SMO（Sequential Minimal Optimization）算法实现了线性 SVM 分类器，
在 iris 数据集上的二分类任务中取得了显著的效果：

✓ 训练集准确率: {train_accuracy:.4f}
✓ 测试集准确率: {test_accuracy:.4f}
✓ 支持向量数: {len(svm_model.support_vectors_idx)}
✓ 决策函数参数:
  - w = {svm_model.w}
  - b = {svm_model.b:.6f}
✓ 间隔宽度: {2.0/np.linalg.norm(svm_model.w):.6f}

SMO 算法通过逐次优化两个拉格朗日乘数，解决 SVM 对偶优化问题，
是一个高效的 SVM 训练方法。
""")

print("\n程序执行完毕！")
plt.show()
