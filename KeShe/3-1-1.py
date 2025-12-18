"""
题目一：采用 scikit-learn 中的线性 SVM 对 iris 数据集进行二分类
（1）选取两个特征和两类数据使用 scikit-learn 中的 SVM 进行二分类
（2）输出：决策边界的参数和截距、支持向量等
（3）可视化：通过散点图可视化数据样本，并画出决策边界和 2 个最大间隔边界，标出支持向量
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 1. 加载 iris 数据集
iris = datasets.load_iris()

# 2. 选取两个特征（花瓣长度和花瓣宽度）和两类数据（类别 1 和类别 2）
# 特征索引：2 = petal length, 3 = petal width
# 类别：1 = versicolor, 2 = virginica
X = iris.data[:, 2:4]  # 选择第3和第4个特征（花瓣长度和花瓣宽度）
y = iris.target

# 只选择类别 1 和类别 2 的数据
mask = (y == 1) | (y == 2)
X = X[mask]
y = y[mask]

print("=" * 60)
print("数据集信息：")
print(f"样本数量: {len(X)}")
print(f"特征名称: petal length (花瓣长度), petal width (花瓣宽度)")
print(f"类别: versicolor (1), virginica (2)")
print("=" * 60)

# 3. 训练线性 SVM 模型
# 使用线性核，C=1.0 是正则化参数
svm_model = SVC(kernel='linear', C=1.0)
svm_model.fit(X, y)

# 4. 输出决策边界的参数和截距、支持向量
print("\n模型参数：")
print(f"决策边界参数 (w): {svm_model.coef_}")
print(f"截距 (b): {svm_model.intercept_}")
print(f"\n支持向量数量: {len(svm_model.support_vectors_)}")
print(f"支持向量索引: {svm_model.support_}")
print(f"\n支持向量坐标：")
for i, sv in enumerate(svm_model.support_vectors_):
    print(f"  支持向量 {i+1}: [{sv[0]:.4f}, {sv[1]:.4f}]")
print("=" * 60)

# 5. 可视化
# 创建网格来绘制决策边界
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

# 计算决策函数值
Z = svm_model.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 绘图
plt.figure(figsize=(10, 8))

# 绘制散点图
plt.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', marker='o', 
            label='Versicolor (类别 1)', s=50, edgecolors='k')
plt.scatter(X[y == 2, 0], X[y == 2, 1], c='green', marker='o', 
            label='Virginica (类别 2)', s=50, edgecolors='k')

# 绘制决策边界（decision_function = 0）
plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black', 
            linestyles='solid')

# 绘制最大间隔边界（decision_function = -1 和 +1）
plt.contour(xx, yy, Z, levels=[-1, 1], linewidths=2, colors='black', 
            linestyles='dashed', alpha=0.8)

# 标出支持向量
plt.scatter(svm_model.support_vectors_[:, 0], 
            svm_model.support_vectors_[:, 1],
            s=200, linewidth=1.5, facecolors='none', 
            edgecolors='red', label='支持向量')

# 设置标签和标题
plt.xlabel('Petal Length (花瓣长度)', fontsize=12)
plt.ylabel('Petal Width (花瓣宽度)', fontsize=12)
plt.title('线性 SVM 二分类 - Iris 数据集', fontsize=14, fontweight='bold')
plt.legend(loc='upper left', fontsize=10)
plt.grid(True, alpha=0.3)

# 添加文本说明
textstr = f'决策边界参数:\nw = [{svm_model.coef_[0][0]:.4f}, {svm_model.coef_[0][1]:.4f}]\nb = {svm_model.intercept_[0]:.4f}\n支持向量数: {len(svm_model.support_vectors_)}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=9,
         verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig('svm_iris_classification.png', dpi=300, bbox_inches='tight')
print("\n图像已保存为 'svm_iris_classification.png'")
plt.show()