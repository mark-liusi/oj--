"""
【讨论二】SVM 中的惩罚系数 C 对模型有何影响？
(1) 尝试改变惩罚系数 C，分析其变化对应间隔宽度、支持向量数量的变化趋势，并解释原因。
(2) 尝试改变惩罚系数 C，分析其对 iris 分类模型性能的影响，并解释原因。
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载数据
iris = datasets.load_iris()
X = iris.data[:, 2:4]  # 花瓣长度和花瓣宽度
y = iris.target
mask = (y == 1) | (y == 2)
X = X[mask]
y = y[mask]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 测试不同的 C 值
C_values = [0.01, 0.1, 1, 10, 100, 1000]

print("=" * 80)
print("【讨论二】SVM 惩罚系数 C 对模型的影响分析")
print("=" * 80)

# 存储结果
results = []

for C in C_values:
    # 训练模型
    svm = SVC(kernel='linear', C=C)
    svm.fit(X_train, y_train)
    
    # 预测
    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # 计算间隔宽度 (margin = 2/||w||)
    w = svm.coef_[0]
    margin_width = 2 / np.linalg.norm(w)
    
    # 支持向量数量
    n_support_vectors = len(svm.support_vectors_)
    
    results.append({
        'C': C,
        'margin_width': margin_width,
        'n_sv': n_support_vectors,
        'accuracy': accuracy,
        'w': w,
        'b': svm.intercept_[0]
    })
    
    print(f"\n--- C = {C} ---")
    print(f"决策边界参数 ||w||: {np.linalg.norm(w):.4f}")
    print(f"间隔宽度 (2/||w||): {margin_width:.4f}")
    print(f"支持向量数量: {n_support_vectors}")
    print(f"训练集准确率: {svm.score(X_train, y_train):.4f}")
    print(f"测试集准确率: {accuracy:.4f}")

print("\n" + "=" * 80)
print("【结果总结】")
print("=" * 80)

# 创建表格
print(f"\n{'C值':<10} {'间隔宽度':<12} {'支持向量数':<12} {'训练准确率':<12} {'测试准确率':<12}")
print("-" * 70)

for i, res in enumerate(results):
    svm_temp = SVC(kernel='linear', C=res['C'])
    svm_temp.fit(X_train, y_train)
    train_acc = svm_temp.score(X_train, y_train)
    print(f"{res['C']:<10} {res['margin_width']:<12.4f} {res['n_sv']:<12} {train_acc:<12.4f} {res['accuracy']:<12.4f}")

# 可视化分析
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

for idx, C in enumerate(C_values):
    ax = axes[idx]
    
    # 训练模型
    svm = SVC(kernel='linear', C=C)
    svm.fit(X_train, y_train)
    
    # 创建网格
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # 绘制训练数据
    ax.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], 
               c='blue', marker='o', label='训练-类别1', s=50, edgecolors='k', alpha=0.7)
    ax.scatter(X_train[y_train == 2, 0], X_train[y_train == 2, 1], 
               c='green', marker='o', label='训练-类别2', s=50, edgecolors='k', alpha=0.7)
    
    # 绘制测试数据
    ax.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], 
               c='blue', marker='s', label='测试-类别1', s=50, edgecolors='k', alpha=0.5)
    ax.scatter(X_test[y_test == 2, 0], X_test[y_test == 2, 1], 
               c='green', marker='s', label='测试-类别2', s=50, edgecolors='k', alpha=0.5)
    
    # 绘制决策边界和间隔
    ax.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')
    ax.contour(xx, yy, Z, levels=[-1, 1], linewidths=2, colors='black', 
               linestyles='dashed', alpha=0.8)
    
    # 标出支持向量
    ax.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1],
               s=200, linewidth=1.5, facecolors='none', edgecolors='red')
    
    # 计算指标
    margin = 2 / np.linalg.norm(svm.coef_[0])
    acc = svm.score(X_test, y_test)
    
    ax.set_xlabel('花瓣长度', fontsize=10)
    ax.set_ylabel('花瓣宽度', fontsize=10)
    ax.set_title(f'C = {C}\n间隔={margin:.4f}, 支持向量={len(svm.support_vectors_)}, 准确率={acc:.4f}', 
                 fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    if idx == 0:
        ax.legend(fontsize=8, loc='upper left')

plt.tight_layout()
plt.savefig('svm_C_comparison.png', dpi=300, bbox_inches='tight')
print("\n图像已保存为 'svm_C_comparison.png'")

# 绘制趋势图
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 间隔宽度趋势
C_list = [r['C'] for r in results]
margin_list = [r['margin_width'] for r in results]
sv_list = [r['n_sv'] for r in results]
acc_list = [r['accuracy'] for r in results]

axes[0].plot(C_list, margin_list, 'o-', linewidth=2, markersize=8, color='blue')
axes[0].set_xlabel('惩罚系数 C', fontsize=12)
axes[0].set_ylabel('间隔宽度 (2/||w||)', fontsize=12)
axes[0].set_title('C 对间隔宽度的影响', fontsize=13, fontweight='bold')
axes[0].set_xscale('log')
axes[0].grid(True, alpha=0.3)

axes[1].plot(C_list, sv_list, 's-', linewidth=2, markersize=8, color='red')
axes[1].set_xlabel('惩罚系数 C', fontsize=12)
axes[1].set_ylabel('支持向量数量', fontsize=12)
axes[1].set_title('C 对支持向量数量的影响', fontsize=13, fontweight='bold')
axes[1].set_xscale('log')
axes[1].grid(True, alpha=0.3)

axes[2].plot(C_list, acc_list, '^-', linewidth=2, markersize=8, color='green')
axes[2].set_xlabel('惩罚系数 C', fontsize=12)
axes[2].set_ylabel('测试集准确率', fontsize=12)
axes[2].set_title('C 对模型性能的影响', fontsize=13, fontweight='bold')
axes[2].set_xscale('log')
axes[2].set_ylim([0.85, 1.05])
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('svm_C_trends.png', dpi=300, bbox_inches='tight')
print("趋势图已保存为 'svm_C_trends.png'")

print("\n" + "=" * 80)
print("【分析与解释】")
print("=" * 80)
print("""
(1) C 对间隔宽度和支持向量数量的影响：

   ✓ 当 C 较小时（如 C=0.01）：
     - 模型容忍更多的错误分类和间隔内的样本
     - 间隔宽度较大（软间隔）
     - 支持向量数量较多
     - 模型更关注最大化间隔，而不是完美分类所有样本
   
   ✓ 当 C 较大时（如 C=100, 1000）：
     - 模型要求更严格，惩罚错误分类更重
     - 间隔宽度较小（接近硬间隔）
     - 支持向量数量较少
     - 模型更关注正确分类样本，可能过拟合

(2) C 对 iris 分类模型性能的影响：

   ✓ 从测试准确率来看：
     - C 在适中范围（0.1 ~ 10）时，模型性能较稳定
     - C 过小可能导致欠拟合（间隔太宽，决策边界不够精确）
     - C 过大可能导致过拟合（过度关注训练数据的细节）
   
   ✓ 最佳 C 值选择：
     - 对于 iris 数据集，C=1 或 C=10 通常效果较好
     - 实际应用中应通过交叉验证选择最优 C 值

【结论】
惩罚系数 C 是 SVM 中控制模型复杂度的关键参数：
- C 控制了模型对错误分类的容忍度
- 需要在间隔最大化和分类准确性之间取得平衡
- 应根据具体数据特点通过交叉验证选择合适的 C 值
""")

plt.show()
print("\n程序执行完毕！")
