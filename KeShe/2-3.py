"""
实验二 - 题目三：采用 scikit-learn 中的 LogisticRegression 逻辑回归模型对非线性数据集进行分类
使用多项式特征扩展来处理非线性可分数据

【讨论二】改变degree参数，观察欠拟合->拟合->过拟合的过程
【讨论四】手动调参寻找最优参数组合
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
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
               marker='o', s=30, label='类别 0')
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c='red', edgecolors='black', 
               marker='s', s=30, label='类别 1')
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)

# ==================== 生成非线性数据集 ====================
X_moons, y_moons = make_moons(n_samples=500, noise=0.2, random_state=42)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X_moons, y_moons, test_size=0.2, random_state=42
)

# ==================== 【讨论二】观察不同degree对模型的影响 ====================
print("=" * 70)
print("【讨论二】改变degree参数，观察欠拟合->拟合->过拟合的过程")
print("=" * 70)

# 测试不同的degree值
degrees = [1, 2, 3, 4, 5, 8, 10, 15]
train_scores = []
test_scores = []

print(f"\n{'Degree':<10} {'训练集准确率':<15} {'测试集准确率':<15} {'状态判断':<15}")
print("-" * 70)

for degree in degrees:
    # 创建pipeline
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(solver='lbfgs', max_iter=1000))
    ])
    
    # 训练模型
    model.fit(X_train, y_train)
    
    # 计算准确率
    train_score = accuracy_score(y_train, model.predict(X_train))
    test_score = accuracy_score(y_test, model.predict(X_test))
    
    train_scores.append(train_score)
    test_scores.append(test_score)
    
    # 判断模型状态
    if train_score < 0.85:
        status = "欠拟合"
    elif train_score - test_score > 0.1:
        status = "过拟合"
    else:
        status = "良好拟合"
    
    print(f"{degree:<10} {train_score:<15.4f} {test_score:<15.4f} {status:<15}")

# 可视化degree对性能的影响
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 左图：准确率随degree变化
ax1.plot(degrees, train_scores, 'o-', label='训练集准确率', linewidth=2, markersize=8)
ax1.plot(degrees, test_scores, 's-', label='测试集准确率', linewidth=2, markersize=8)
ax1.set_xlabel('Degree (多项式阶数)', fontsize=12)
ax1.set_ylabel('准确率', fontsize=12)
ax1.set_title('不同Degree对模型性能的影响', fontsize=14)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(degrees)

# 右图：过拟合程度（训练集与测试集准确率差异）
overfitting = np.array(train_scores) - np.array(test_scores)
ax2.plot(degrees, overfitting, 'ro-', linewidth=2, markersize=8)
ax2.axhline(y=0, color='green', linestyle='--', label='理想状态')
ax2.axhline(y=0.1, color='orange', linestyle='--', label='过拟合阈值')
ax2.set_xlabel('Degree (多项式阶数)', fontsize=12)
ax2.set_ylabel('训练集 - 测试集准确率', fontsize=12)
ax2.set_title('过拟合程度分析', fontsize=14)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xticks(degrees)

plt.tight_layout()
plt.savefig('degree_analysis.png', dpi=300, bbox_inches='tight')
print("\n图片已保存为 'degree_analysis.png'")

# 可视化不同degree的决策边界
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.ravel()

selected_degrees = [1, 2, 3, 4, 5, 8, 10, 15]
for idx, degree in enumerate(selected_degrees):
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(solver='lbfgs', max_iter=1000))
    ])
    model.fit(X_train, y_train)
    
    train_score = accuracy_score(y_train, model.predict(X_train))
    test_score = accuracy_score(y_test, model.predict(X_test))
    
    title = f'Degree={degree}\n训练:{train_score:.3f} 测试:{test_score:.3f}'
    plot_decision_boundary(axes[idx], model, X_moons, y_moons, title)

plt.tight_layout()
plt.savefig('degree_decision_boundaries.png', dpi=300, bbox_inches='tight')
print("图片已保存为 'degree_decision_boundaries.png'")

# ==================== 【讨论四】手动调参寻找最优参数 ====================
print("\n" + "=" * 70)
print("【讨论四】手动调参寻找最优参数组合")
print("=" * 70)

# 步骤1：固定正则化类型和C，寻找最优degree
print("\n【步骤1】固定正则化(L2)和C=1.0，寻找最优degree")
penalty = 'l2'
C_fixed = 1.0
degree_range = range(1, 16)

best_degree = None
best_test_score = 0
degree_results = []

print(f"\n{'Degree':<10} {'训练集':<12} {'测试集':<12}")
print("-" * 40)

for degree in degree_range:
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(penalty=penalty, C=C_fixed, solver='lbfgs', max_iter=2000))
    ])
    model.fit(X_train, y_train)
    
    train_score = accuracy_score(y_train, model.predict(X_train))
    test_score = accuracy_score(y_test, model.predict(X_test))
    degree_results.append((degree, train_score, test_score))
    
    print(f"{degree:<10} {train_score:<12.4f} {test_score:<12.4f}")
    
    if test_score > best_test_score:
        best_test_score = test_score
        best_degree = degree

print(f"\n最优degree: {best_degree}, 测试集准确率: {best_test_score:.4f}")

# 步骤2：固定最优degree和正则化类型，寻找最优C
print(f"\n【步骤2】固定degree={best_degree}和正则化(L2)，寻找最优C")
C_range = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]

best_C = None
best_test_score_C = 0
C_results = []

print(f"\n{'C值':<12} {'训练集':<12} {'测试集':<12}")
print("-" * 40)

for C in C_range:
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=best_degree)),
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(penalty=penalty, C=C, solver='lbfgs', max_iter=2000))
    ])
    model.fit(X_train, y_train)
    
    train_score = accuracy_score(y_train, model.predict(X_train))
    test_score = accuracy_score(y_test, model.predict(X_test))
    C_results.append((C, train_score, test_score))
    
    print(f"{C:<12.3f} {train_score:<12.4f} {test_score:<12.4f}")
    
    if test_score > best_test_score_C:
        best_test_score_C = test_score
        best_C = C

print(f"\n最优C: {best_C}, 测试集准确率: {best_test_score_C:.4f}")

# 步骤3：尝试不同正则化类型
print(f"\n【步骤3】固定degree={best_degree}和C={best_C}，比较不同正则化类型")
penalties = ['l1', 'l2']
penalty_results = []

print(f"\n{'正则化':<12} {'训练集':<12} {'测试集':<12}")
print("-" * 40)

for pen in penalties:
    solver = 'saga' if pen == 'l1' else 'lbfgs'
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=best_degree)),
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(penalty=pen, C=best_C, solver=solver, max_iter=2000))
    ])
    model.fit(X_train, y_train)
    
    train_score = accuracy_score(y_train, model.predict(X_train))
    test_score = accuracy_score(y_test, model.predict(X_test))
    penalty_results.append((pen, train_score, test_score))
    
    print(f"{pen:<12} {train_score:<12.4f} {test_score:<12.4f}")

# 如果L1更好，则用L1重新寻优C
best_penalty = max(penalty_results, key=lambda x: x[2])[0]
print(f"\n最优正则化类型: {best_penalty}")

if best_penalty == 'l1':
    print(f"\n【步骤4】使用L1正则化重新寻找最优C")
    best_C_l1 = None
    best_test_score_l1 = 0
    
    print(f"\n{'C值':<12} {'训练集':<12} {'测试集':<12}")
    print("-" * 40)
    
    for C in C_range:
        model = Pipeline([
            ('poly', PolynomialFeatures(degree=best_degree)),
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(penalty='l1', C=C, solver='saga', max_iter=2000))
        ])
        model.fit(X_train, y_train)
        
        train_score = accuracy_score(y_train, model.predict(X_train))
        test_score = accuracy_score(y_test, model.predict(X_test))
        
        print(f"{C:<12.3f} {train_score:<12.4f} {test_score:<12.4f}")
        
        if test_score > best_test_score_l1:
            best_test_score_l1 = test_score
            best_C_l1 = C
    
    print(f"\nL1正则化最优C: {best_C_l1}, 测试集准确率: {best_test_score_l1:.4f}")
    final_C = best_C_l1
    final_score = best_test_score_l1
else:
    final_C = best_C
    final_score = best_test_score_C

# 最终结果
print("\n" + "=" * 70)
print("【最终最优参数组合】")
print("=" * 70)
print(f"Degree: {best_degree}")
print(f"正则化类型: {best_penalty}")
print(f"正则化系数C: {final_C}")
print(f"测试集准确率: {final_score:.4f}")

# 可视化调参结果
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 子图1: Degree对性能的影响
degrees_list = [x[0] for x in degree_results]
train_list = [x[1] for x in degree_results]
test_list = [x[2] for x in degree_results]

axes[0].plot(degrees_list, train_list, 'o-', label='训练集', linewidth=2)
axes[0].plot(degrees_list, test_list, 's-', label='测试集', linewidth=2)
axes[0].axvline(x=best_degree, color='r', linestyle='--', label=f'最优degree={best_degree}')
axes[0].set_xlabel('Degree', fontsize=12)
axes[0].set_ylabel('准确率', fontsize=12)
axes[0].set_title('Degree调优结果', fontsize=14)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 子图2: C对性能的影响
C_list = [x[0] for x in C_results]
train_C_list = [x[1] for x in C_results]
test_C_list = [x[2] for x in C_results]

axes[1].semilogx(C_list, train_C_list, 'o-', label='训练集', linewidth=2)
axes[1].semilogx(C_list, test_C_list, 's-', label='测试集', linewidth=2)
axes[1].axvline(x=final_C, color='r', linestyle='--', label=f'最优C={final_C}')
axes[1].set_xlabel('正则化系数 C (log scale)', fontsize=12)
axes[1].set_ylabel('准确率', fontsize=12)
axes[1].set_title('C调优结果', fontsize=14)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 子图3: 最优模型的决策边界
solver_final = 'saga' if best_penalty == 'l1' else 'lbfgs'
final_model = Pipeline([
    ('poly', PolynomialFeatures(degree=best_degree)),
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(penalty=best_penalty, C=final_C, solver=solver_final, max_iter=2000))
])
final_model.fit(X_train, y_train)

title = f'最优模型决策边界\nDegree={best_degree}, {best_penalty}, C={final_C}\n测试准确率={final_score:.4f}'
plot_decision_boundary(axes[2], final_model, X_moons, y_moons, title)

plt.tight_layout()
plt.savefig('hyperparameter_tuning.png', dpi=300, bbox_inches='tight')
print("\n图片已保存为 'hyperparameter_tuning.png'")

plt.show()
