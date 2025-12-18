import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import warnings

warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 绘制决策边界函数 ====================
def plot_decision_boundary(ax, model, X, y, title):
    """绘制决策边界"""
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                          np.arange(y_min, y_max, 0.02))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    ax.scatter(X[y == 0, 0], X[y == 0, 1], c='blue', edgecolors='black', 
               marker='o', s=50, label='类别 0')
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c='red', edgecolors='black', 
               marker='s', s=50, label='类别 1')
    ax.set_title(title)
    ax.legend()

# ==================== (1) 生成非线性数据集 ====================
# 数据集1: 月牙形数据 (make_moons)
X_moons, y_moons = make_moons(n_samples=500, noise=0.2, random_state=42)

# 数据集2: 同心圆数据 (make_circles)
X_circles, y_circles = make_circles(n_samples=500, noise=0.1, factor=0.5, random_state=42)

# ==================== (2) 划分数据集 ====================
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_moons, y_moons, test_size=0.2, random_state=42
)

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_circles, y_circles, test_size=0.2, random_state=42
)

# ==================== (3) 定义模型 ====================
# 线性逻辑回归 (不使用多项式特征)
def create_linear_model():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(solver='lbfgs', C=1.0, max_iter=1000))
    ])

# 多项式逻辑回归 (使用多项式特征扩展)
def create_poly_model(degree=2):
    return Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(solver='lbfgs', C=1.0, max_iter=1000))
    ])

# ==================== (4) 在月牙形数据上训练和评估 ====================
# 线性模型
model_linear_m = create_linear_model()
model_linear_m.fit(X_train_m, y_train_m)
acc_linear_m = accuracy_score(y_test_m, model_linear_m.predict(X_test_m))

# 多项式模型 (degree=2)
model_poly2_m = create_poly_model(degree=2)
model_poly2_m.fit(X_train_m, y_train_m)
acc_poly2_m = accuracy_score(y_test_m, model_poly2_m.predict(X_test_m))

# 多项式模型 (degree=3)
model_poly3_m = create_poly_model(degree=3)
model_poly3_m.fit(X_train_m, y_train_m)
acc_poly3_m = accuracy_score(y_test_m, model_poly3_m.predict(X_test_m))

# ==================== (5) 在同心圆数据上训练和评估 ====================
# 线性模型
model_linear_c = create_linear_model()
model_linear_c.fit(X_train_c, y_train_c)
acc_linear_c = accuracy_score(y_test_c, model_linear_c.predict(X_test_c))

# 多项式模型 (degree=2)
model_poly2_c = create_poly_model(degree=2)
model_poly2_c.fit(X_train_c, y_train_c)
acc_poly2_c = accuracy_score(y_test_c, model_poly2_c.predict(X_test_c))

# 多项式模型 (degree=3)
model_poly3_c = create_poly_model(degree=3)
model_poly3_c.fit(X_train_c, y_train_c)
acc_poly3_c = accuracy_score(y_test_c, model_poly3_c.predict(X_test_c))

# ==================== (6) 模型对比总结 ====================
print("【模型对比总结】")
print(f"{'数据集':<15} {'线性模型':<12} {'多项式(d=2)':<12} {'多项式(d=3)':<12}")
print("-" * 55)
print(f"{'月牙形':<15} {acc_linear_m:<12.4f} {acc_poly2_m:<12.4f} {acc_poly3_m:<12.4f}")
print(f"{'同心圆':<15} {acc_linear_c:<12.4f} {acc_poly2_c:<12.4f} {acc_poly3_c:<12.4f}")

# ==================== (7) 可视化 ====================
fig, axes = plt.subplots(2, 4, figsize=(18, 9))

# 第一行: 月牙形数据
axes[0, 0].scatter(X_moons[y_moons == 0, 0], X_moons[y_moons == 0, 1], 
                   c='blue', edgecolors='black', marker='o', s=30, label='类别 0')
axes[0, 0].scatter(X_moons[y_moons == 1, 0], X_moons[y_moons == 1, 1], 
                   c='red', edgecolors='black', marker='s', s=30, label='类别 1')
axes[0, 0].set_title('月牙形数据集')
axes[0, 0].legend()

plot_decision_boundary(axes[0, 1], model_linear_m, X_moons, y_moons, 
                       f'线性模型\n准确率: {acc_linear_m:.4f}')
plot_decision_boundary(axes[0, 2], model_poly2_m, X_moons, y_moons, 
                       f'多项式(d=2)\n准确率: {acc_poly2_m:.4f}')
plot_decision_boundary(axes[0, 3], model_poly3_m, X_moons, y_moons, 
                       f'多项式(d=3)\n准确率: {acc_poly3_m:.4f}')

# 第二行: 同心圆数据
axes[1, 0].scatter(X_circles[y_circles == 0, 0], X_circles[y_circles == 0, 1], 
                   c='blue', edgecolors='black', marker='o', s=30, label='类别 0')
axes[1, 0].scatter(X_circles[y_circles == 1, 0], X_circles[y_circles == 1, 1], 
                   c='red', edgecolors='black', marker='s', s=30, label='类别 1')
axes[1, 0].set_title('同心圆数据集')
axes[1, 0].legend()

plot_decision_boundary(axes[1, 1], model_linear_c, X_circles, y_circles, 
                       f'线性模型\n准确率: {acc_linear_c:.4f}')
plot_decision_boundary(axes[1, 2], model_poly2_c, X_circles, y_circles, 
                       f'多项式(d=2)\n准确率: {acc_poly2_c:.4f}')
plot_decision_boundary(axes[1, 3], model_poly3_c, X_circles, y_circles, 
                       f'多项式(d=3)\n准确率: {acc_poly3_c:.4f}')

plt.tight_layout()
plt.savefig('nonlinear_classification.png', dpi=300, bbox_inches='tight')
plt.show()
