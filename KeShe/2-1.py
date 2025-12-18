import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import warnings

warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== (1) 导入数据 ====================
# 加载 iris 数据集
iris = load_iris()
X = iris.data
y = iris.target
mask = y != 0  # 排除类别 0
X_binary = X[mask]
y_binary = y[mask]
y_binary = y_binary - 1  # 将标签改为 0 和 1
# ==================== (2) 划分数据集 ====================

X_train, X_test, y_train, y_test = train_test_split(
    X_binary, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

# ==================== (3) 数据标准化 ====================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==================== (4) 训练模型 ====================
# 创建逻辑回归模型
model = LogisticRegression(
    solver='lbfgs',      # 优化求解器
    penalty='l2',        # L2 正则化
    C=1.0,               # 正则化强度的倒数
    max_iter=1000,       # 最大迭代次数
    random_state=42
)

# 训练模型
model.fit(X_train_scaled, y_train)

# ==================== (5) 模型预测与评估 ====================

# 预测
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# 预测概率
y_test_prob = model.predict_proba(X_test_scaled)

# 计算评估指标
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)

print(f"训练集准确率: {train_acc:.4f}")
print(f"测试集准确率: {test_acc:.4f}")
print(f"测试集精确率 (Precision): {test_precision:.4f}")
print(f"测试集召回率 (Recall): {test_recall:.4f}")
print(f"测试集 F1 值: {test_f1:.4f}")

# 混淆矩阵
print("\n混淆矩阵:")
cm = confusion_matrix(y_test, y_test_pred)
print(cm)
print(f"  TN={cm[0,0]}, FP={cm[0,1]}")
print(f"  FN={cm[1,0]}, TP={cm[1,1]}")

# 分类报告
print("\n分类报告:")
print(classification_report(y_test, y_test_pred, 
                            target_names=['versicolor', 'virginica']))

# ==================== (6) 可视化 ====================
# 创建图形
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 图1: 混淆矩阵热力图
im = axes[0].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
axes[0].figure.colorbar(im, ax=axes[0])
axes[0].set(xticks=[0, 1], yticks=[0, 1],
            xticklabels=['versicolor', 'virginica'],
            yticklabels=['versicolor', 'virginica'],
            xlabel='预测类别', ylabel='真实类别',
            title='混淆矩阵')

# 在格子中显示数值
for i in range(2):
    for j in range(2):
        axes[0].text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max()/2 else "black",
                    fontsize=16)

# 图2: 使用前两个特征绘制决策边界
# 只使用前两个特征进行可视化
X_vis = X_binary[:, :2]
X_train_vis, X_test_vis, y_train_vis, y_test_vis = train_test_split(
    X_vis, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

scaler_vis = StandardScaler()
X_train_vis_scaled = scaler_vis.fit_transform(X_train_vis)
X_test_vis_scaled = scaler_vis.transform(X_test_vis)

# 训练用于可视化的模型
model_vis = LogisticRegression(solver='lbfgs', C=1.0, random_state=42)
model_vis.fit(X_train_vis_scaled, y_train_vis)

# 绘制决策边界
x_min, x_max = X_train_vis_scaled[:, 0].min() - 1, X_train_vis_scaled[:, 0].max() + 1
y_min, y_max = X_train_vis_scaled[:, 1].min() - 1, X_train_vis_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                      np.arange(y_min, y_max, 0.02))

Z = model_vis.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

axes[1].contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
axes[1].scatter(X_train_vis_scaled[:, 0], X_train_vis_scaled[:, 1], 
                c=y_train_vis, cmap=plt.cm.RdYlBu, edgecolors='black', 
                marker='o', s=80, label='训练集')
axes[1].scatter(X_test_vis_scaled[:, 0], X_test_vis_scaled[:, 1], 
                c=y_test_vis, cmap=plt.cm.RdYlBu, edgecolors='black', 
                marker='^', s=100, label='测试集')
axes[1].set_xlabel(f'{iris.feature_names[0]} (标准化)')
axes[1].set_ylabel(f'{iris.feature_names[1]} (标准化)')
axes[1].set_title('决策边界 (前两个特征)')
axes[1].legend()

plt.tight_layout()
plt.savefig('iris_binary_classification.png', dpi=300, bbox_inches='tight')
plt.show()