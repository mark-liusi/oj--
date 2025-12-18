import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== (1) 导入数据 ====================
iris = load_iris()
X = iris.data
y = iris.target

# ==================== (2) 划分数据集 ====================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==================== (3) 数据标准化 ====================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==================== (4) 训练多分类模型 ====================
# 模型1: OvR (One-vs-Rest) 策略
model_ovr = LogisticRegression(
    solver='lbfgs',
    multi_class='ovr',  # One-vs-Rest
    C=1.0,
    max_iter=1000,
    random_state=42
)
model_ovr.fit(X_train_scaled, y_train)

# 模型2: Multinomial (Softmax) 策略
model_multinomial = LogisticRegression(
    solver='lbfgs',
    multi_class='multinomial',  # Softmax 回归
    C=1.0,
    max_iter=1000,
    random_state=42
)
model_multinomial.fit(X_train_scaled, y_train)

# 模型3: 使用 OneVsOneClassifier 元估计器
base_model = LogisticRegression(solver='lbfgs', C=1.0, max_iter=1000, random_state=42)
model_ovo = OneVsOneClassifier(base_model)
model_ovo.fit(X_train_scaled, y_train)

# ==================== (5) 模型预测与评估 ====================

models = {
    'OvR': model_ovr,
    'Multinomial (Softmax)': model_multinomial,
    'OvO': model_ovo
}

results = {}

for name, model in models.items():
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    results[name] = {
        'train_acc': train_acc,
        'test_acc': test_acc,
        'y_test_pred': y_test_pred
    }

y_pred_best = results['Multinomial (Softmax)']['y_test_pred']
cm = confusion_matrix(y_test, y_pred_best)

# ==================== (6) 模型对比 ====================
print("模型评估结果:")
print(f"{'模型':<25} {'训练集准确率':<15} {'测试集准确率':<15}")
for name, res in results.items():
    print(f"{name:<25} {res['train_acc']:<15.4f} {res['test_acc']:<15.4f}")

# ==================== (7) 可视化 ====================

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 图1: 各模型准确率对比
ax1 = axes[0, 0]
model_names = list(results.keys())
train_accs = [results[n]['train_acc'] for n in model_names]
test_accs = [results[n]['test_acc'] for n in model_names]

x = np.arange(len(model_names))
width = 0.35

bars1 = ax1.bar(x - width/2, train_accs, width, label='训练集', color='steelblue')
bars2 = ax1.bar(x + width/2, test_accs, width, label='测试集', color='coral')

ax1.set_ylabel('准确率')
ax1.set_title('多分类策略准确率对比')
ax1.set_xticks(x)
ax1.set_xticklabels(model_names, rotation=15)
ax1.legend()
ax1.set_ylim(0.8, 1.05)

# 在柱子上显示数值
for bar in bars1:
    ax1.annotate(f'{bar.get_height():.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha='center', va='bottom', fontsize=9)
for bar in bars2:
    ax1.annotate(f'{bar.get_height():.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha='center', va='bottom', fontsize=9)

# 图2: Multinomial 模型混淆矩阵
ax2 = axes[0, 1]
im = ax2.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax2.figure.colorbar(im, ax=ax2)
ax2.set(xticks=[0, 1, 2], yticks=[0, 1, 2],
        xticklabels=iris.target_names,
        yticklabels=iris.target_names,
        xlabel='预测类别', ylabel='真实类别',
        title='混淆矩阵 (Multinomial)')

for i in range(3):
    for j in range(3):
        ax2.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > cm.max()/2 else "black",
                fontsize=14)

# 图3 & 图4: 决策边界 (使用前两个特征)
X_vis = X[:, :2]
X_train_vis, X_test_vis, y_train_vis, y_test_vis = train_test_split(
    X_vis, y, test_size=0.2, random_state=42, stratify=y
)

scaler_vis = StandardScaler()
X_train_vis_scaled = scaler_vis.fit_transform(X_train_vis)
X_test_vis_scaled = scaler_vis.transform(X_test_vis)

# 训练用于可视化的模型
model_ovr_vis = LogisticRegression(solver='lbfgs', multi_class='ovr', max_iter=1000, random_state=42)
model_multi_vis = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=1000, random_state=42)
model_ovr_vis.fit(X_train_vis_scaled, y_train_vis)
model_multi_vis.fit(X_train_vis_scaled, y_train_vis)

# 绘制决策边界函数
def plot_decision_boundary(ax, model, X, y, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                          np.arange(y_min, y_max, 0.02))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, 
                         edgecolors='black', s=50)
    ax.set_xlabel(f'{iris.feature_names[0]} (标准化)')
    ax.set_ylabel(f'{iris.feature_names[1]} (标准化)')
    ax.set_title(title)
    return scatter

# 图3: OvR 决策边界
plot_decision_boundary(axes[1, 0], model_ovr_vis, X_train_vis_scaled, y_train_vis, 
                       'OvR 决策边界 (前两个特征)')

# 图4: Multinomial 决策边界
scatter = plot_decision_boundary(axes[1, 1], model_multi_vis, X_train_vis_scaled, y_train_vis, 
                                  'Multinomial 决策边界 (前两个特征)')

# 添加图例
legend_labels = [iris.target_names[i] for i in range(3)]
axes[1, 1].legend(handles=scatter.legend_elements()[0], labels=legend_labels, 
                   loc='upper left')

plt.tight_layout()
plt.savefig('iris_multiclass_classification.png', dpi=300, bbox_inches='tight')
plt.show()