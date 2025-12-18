import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings

warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 逻辑回归类实现 ====================
class LogisticRegressionNumpy:
    """使用numpy手动实现的逻辑回归"""
    
    def __init__(self, learning_rate=0.01, n_iterations=1000, add_bias=True):
        """
        参数:
            learning_rate: 学习率
            n_iterations: 迭代次数
            add_bias: 是否添加偏置项
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.add_bias = add_bias
        self.weights = None
        self.bias = None
        self.losses = []  # 记录损失值
        
    def _sigmoid(self, z):
        """Sigmoid激活函数"""
        # 防止溢出
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def _add_bias_feature(self, X):
        """添加偏置特征"""
        if self.add_bias:
            return np.c_[np.ones((X.shape[0], 1)), X]
        return X
    
    def _compute_loss(self, y_true, y_pred):
        """计算交叉熵损失"""
        # 防止log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss
    
    def fit(self, X, y):
        """训练模型"""
        # 数据准备
        n_samples, n_features = X.shape
        
        if self.add_bias:
            X = self._add_bias_feature(X)
            n_features += 1
        
        # 初始化权重
        self.weights = np.zeros(n_features)
        
        # 梯度下降
        for i in range(self.n_iterations):
            # 前向传播
            linear_output = np.dot(X, self.weights)
            y_pred = self._sigmoid(linear_output)
            
            # 计算损失
            loss = self._compute_loss(y, y_pred)
            self.losses.append(loss)
            
            # 计算梯度
            gradient = np.dot(X.T, (y_pred - y)) / n_samples
            
            # 更新权重
            self.weights -= self.learning_rate * gradient
            
            # 每100次迭代打印一次
            if (i + 1) % 100 == 0:
                accuracy = np.mean((y_pred >= 0.5).astype(int) == y)
                print(f"迭代 {i+1}/{self.n_iterations} - 损失: {loss:.4f}, 准确率: {accuracy:.4f}")
        
        return self
    
    def predict_proba(self, X):
        """预测概率"""
        if self.add_bias:
            X = self._add_bias_feature(X)
        
        linear_output = np.dot(X, self.weights)
        return self._sigmoid(linear_output)
    
    def predict(self, X):
        """预测类别"""
        return (self.predict_proba(X) >= 0.5).astype(int)
    
    def score(self, X, y):
        """计算准确率"""
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

# ==================== 加载和预处理数据 ====================
print("=" * 70)
print("题目四：使用 numpy 手动实现逻辑回归对 iris 数据进行二分类")
print("=" * 70)

# 加载数据
iris = load_iris()
X = iris.data
y = iris.target

# 选择两个类别进行二分类 (versicolor vs virginica)
# 类别1和类别2
mask = (y == 1) | (y == 2)
X_binary = X[mask]
y_binary = y[mask]
y_binary = (y_binary == 2).astype(int)  # 转换为0/1标签

print(f"\n数据集信息:")
print(f"样本数量: {X_binary.shape[0]}")
print(f"特征数量: {X_binary.shape[1]}")
print(f"类别0 (versicolor) 数量: {np.sum(y_binary == 0)}")
print(f"类别1 (virginica) 数量: {np.sum(y_binary == 1)}")

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_binary, y_binary, test_size=0.3, random_state=42, stratify=y_binary
)

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n训练集样本数: {X_train.shape[0]}")
print(f"测试集样本数: {X_test.shape[0]}")

# ==================== 训练自定义逻辑回归模型 ====================
print("\n" + "=" * 70)
print("训练自定义逻辑回归模型")
print("=" * 70)

# 创建并训练模型
model = LogisticRegressionNumpy(learning_rate=0.1, n_iterations=1000, add_bias=True)
model.fit(X_train_scaled, y_train)

# ==================== 模型评估 ====================
print("\n" + "=" * 70)
print("模型评估")
print("=" * 70)

# 训练集评估
train_pred = model.predict(X_train_scaled)
train_acc = accuracy_score(y_train, train_pred)
print(f"\n训练集准确率: {train_acc:.4f}")

# 测试集评估
test_pred = model.predict(X_test_scaled)
test_acc = accuracy_score(y_test, test_pred)
print(f"测试集准确率: {test_acc:.4f}")

# 混淆矩阵
print("\n混淆矩阵:")
cm = confusion_matrix(y_test, test_pred)
print(cm)

# 详细分类报告
print("\n分类报告:")
print(classification_report(y_test, test_pred, 
                          target_names=['versicolor', 'virginica']))

# 打印模型权重
print("\n模型参数:")
if model.add_bias:
    print(f"偏置项: {model.weights[0]:.4f}")
    print(f"权重: {model.weights[1:]}")
else:
    print(f"权重: {model.weights}")

# ==================== 与sklearn对比 ====================
print("\n" + "=" * 70)
print("与 sklearn 的 LogisticRegression 对比")
print("=" * 70)

from sklearn.linear_model import LogisticRegression as SklearnLR

sklearn_model = SklearnLR(max_iter=1000, random_state=42)
sklearn_model.fit(X_train_scaled, y_train)

sklearn_train_acc = sklearn_model.score(X_train_scaled, y_train)
sklearn_test_acc = sklearn_model.score(X_test_scaled, y_test)

print(f"\nSklearn模型 - 训练集准确率: {sklearn_train_acc:.4f}")
print(f"Sklearn模型 - 测试集准确率: {sklearn_test_acc:.4f}")

print("\n模型对比:")
print(f"{'模型':<20} {'训练集准确率':<15} {'测试集准确率':<15}")
print("-" * 50)
print(f"{'自定义Numpy实现':<20} {train_acc:<15.4f} {test_acc:<15.4f}")
print(f"{'Sklearn实现':<20} {sklearn_train_acc:<15.4f} {sklearn_test_acc:<15.4f}")

# ==================== 可视化 ====================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 子图1: 损失曲线
axes[0, 0].plot(model.losses, linewidth=2)
axes[0, 0].set_xlabel('迭代次数', fontsize=12)
axes[0, 0].set_ylabel('交叉熵损失', fontsize=12)
axes[0, 0].set_title('训练过程中的损失变化', fontsize=14)
axes[0, 0].grid(True, alpha=0.3)

# 子图2: 预测概率分布
test_proba = model.predict_proba(X_test_scaled)
axes[0, 1].hist(test_proba[y_test == 0], bins=20, alpha=0.6, label='versicolor (真实标签0)', color='blue')
axes[0, 1].hist(test_proba[y_test == 1], bins=20, alpha=0.6, label='virginica (真实标签1)', color='red')
axes[0, 1].axvline(x=0.5, color='green', linestyle='--', linewidth=2, label='分类阈值')
axes[0, 1].set_xlabel('预测概率', fontsize=12)
axes[0, 1].set_ylabel('样本数量', fontsize=12)
axes[0, 1].set_title('测试集预测概率分布', fontsize=14)
axes[0, 1].legend()

# 子图3: 混淆矩阵热力图
im = axes[1, 0].imshow(cm, cmap='Blues', interpolation='nearest')
axes[1, 0].set_xticks([0, 1])
axes[1, 0].set_yticks([0, 1])
axes[1, 0].set_xticklabels(['versicolor', 'virginica'])
axes[1, 0].set_yticklabels(['versicolor', 'virginica'])
axes[1, 0].set_xlabel('预测标签', fontsize=12)
axes[1, 0].set_ylabel('真实标签', fontsize=12)
axes[1, 0].set_title('混淆矩阵', fontsize=14)

# 在矩阵中添加数字
for i in range(2):
    for j in range(2):
        text = axes[1, 0].text(j, i, cm[i, j], ha="center", va="center", 
                              color="white" if cm[i, j] > cm.max()/2 else "black",
                              fontsize=20, fontweight='bold')

plt.colorbar(im, ax=axes[1, 0])

# 子图4: 决策边界 (使用前两个特征)
if X_train_scaled.shape[1] >= 2:
    # 使用前两个特征训练简化模型用于可视化
    model_2d = LogisticRegressionNumpy(learning_rate=0.1, n_iterations=1000, add_bias=True)
    model_2d.fit(X_train_scaled[:, :2], y_train)
    
    # 创建网格
    h = 0.02
    x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
    y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # 预测
    Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # 绘制决策边界
    axes[1, 1].contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    
    # 绘制训练数据
    scatter = axes[1, 1].scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], 
                                c=y_train, cmap=plt.cm.RdYlBu, edgecolors='black', s=50)
    axes[1, 1].set_xlabel('特征1 (标准化后)', fontsize=12)
    axes[1, 1].set_ylabel('特征2 (标准化后)', fontsize=12)
    axes[1, 1].set_title('决策边界 (使用前两个特征)', fontsize=14)
    
    # 添加图例
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor='#1f77b4', markersize=10, label='versicolor'),
                      plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor='#d62728', markersize=10, label='virginica')]
    axes[1, 1].legend(handles=legend_elements)

plt.tight_layout()
plt.savefig('numpy_logistic_regression.png', dpi=300, bbox_inches='tight')
print("\n图片已保存为 'numpy_logistic_regression.png'")

plt.show()

print("\n" + "=" * 70)
print("实验完成！")
print("=" * 70)
