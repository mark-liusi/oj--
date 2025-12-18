import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# ==================== (1) 导入数据 ====================
# 数据文件路径
datafile = r'D:\\python-3.8\\Lib\\site-packages\\sklearn\\datasets\\data\\boston_house_prices.csv'

try:
    # 读取 csv 文件
    with open(datafile, encoding='utf-8') as f:
        data = np.loadtxt(f, delimiter=',', skiprows=2)
    # 分离特征和标签
    X = data[:, :-1]  # 前13列是特征
    y = data[:, -1]   # 最后一列是标签（房价）
    
except FileNotFoundError:
    print(f"文件未找到，使用 sklearn 的 fetch_openml 加载数据...")
    from sklearn.datasets import fetch_openml
    
    boston = fetch_openml(name='boston', version=1, parser='auto')
    X = boston.data.values if hasattr(boston.data, 'values') else boston.data
    y = boston.target.values if hasattr(boston.target, 'values') else boston.target

# ==================== (2) 划分数据集 ====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"训练集样本数: {X_train.shape[0]}")
print(f"测试集样本数: {X_test.shape[0]}")

# ==================== (3) 数据归一化 ====================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# ==================== (4) 训练模型 model(train_x, train_y) ====================

def model(train_x, train_y, learning_rate=0.01, epochs=1000):
    """
    使用批量梯度下降法训练线性回归模型
    
    参数:
        train_x: 训练特征数据
        train_y: 训练标签数据
        learning_rate: 学习率
        epochs: 迭代次数
    
    返回:
        w: 权重参数
        loss_history: 损失函数历史记录
    """
    
    # (a) 初始化参数 w
    m, n = train_x.shape  # m 个样本，n 个特征
    w = np.zeros(n + 1)   # n 个权重 + 1 个偏置项
    
    # 将偏置项合并到 X 中（在 X 前面添加一列全为 1 的列）
    X = np.concatenate([np.ones((m, 1)), train_x], axis=1)
    y = train_y
    
    loss_history = []  # 记录每次迭代的损失值
    
    # 开始迭代训练
    for epoch in range(epochs):
        # (b) 求 f(x) = X * w
        h = np.dot(X, w)
        
        # (c) 求 J(w) - 计算损失函数（均方误差）
        m_samples = len(X)
        loss = np.sum((np.dot(X, w) - y) ** 2) / (2 * m_samples)
        loss_history.append(loss)
        
        # (d) 求梯度
        diff = h - y
        
        # (e) 更新参数 w
        # w[j] = w[j] - learning_rate / X.shape[0] * np.sum(diff * X[:, j])
        for j, theta in enumerate(w):
            w[j] = theta - learning_rate / X.shape[0] * np.sum(diff * X[:, j])
        
    
    print(f"训练完成！最终损失: {loss_history[-1]:.4f}")
    
    return w, loss_history

# 训练模型
learning_rate = 0.01
epochs = 1000

w, loss_history = model(X_train_scaled, y_train, learning_rate=learning_rate, epochs=epochs)

# ==================== (5) 画出损失函数随迭代次数的变化曲线 ====================

plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), loss_history, 'b-', linewidth=2)
plt.xlabel('迭代次数 (Epochs)', fontsize=12)
plt.ylabel('损失函数 J(w)', fontsize=12)
plt.title('批量梯度下降 - 损失函数变化曲线', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('1-2-losscurve.png', dpi=300, bbox_inches='tight')
plt.show()

# ==================== (6) 测试集数据进行预测，模型评估 ====================

# 预测函数
def predict(X, w):
    """使用训练好的参数进行预测"""
    m = X.shape[0]
    X_with_bias = np.concatenate([np.ones((m, 1)), X], axis=1)
    return np.dot(X_with_bias, w)

# 训练集预测
y_train_pred = predict(X_train_scaled, w)
train_mse = mean_squared_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

# 测试集预测
y_test_pred = predict(X_test_scaled, w)
test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("\n模型评估结果:")
print(f"训练集 MSE: {train_mse:.4f}, R²: {train_r2:.4f}")
print(f"测试集 MSE: {test_mse:.4f}, R²: {test_r2:.4f}")

# ==================== (7) 可视化：展示数据拟合的效果 ====================

# 创建图形
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# 训练集拟合效果
axes[0].scatter(y_train, y_train_pred, alpha=0.5, s=30)
axes[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 
             'r--', lw=2, label='完美拟合线')
axes[0].set_xlabel('真实房价', fontsize=12)
axes[0].set_ylabel('预测房价', fontsize=12)
axes[0].set_title(f'训练集拟合效果\nMSE={train_mse:.2f}, R^2={train_r2:.4f}', 
                  fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 测试集拟合效果
axes[1].scatter(y_test, y_test_pred, alpha=0.5, s=30, color='orange')
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', lw=2, label='完美拟合线')
axes[1].set_xlabel('真实房价', fontsize=12)
axes[1].set_ylabel('预测房价', fontsize=12)
axes[1].set_title(f'测试集拟合效果\nMSE={test_mse:.2f}, R^2={test_r2:.4f}', 
                  fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('1-2-fittingresult.png', dpi=300, bbox_inches='tight')
plt.show()
