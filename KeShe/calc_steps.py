import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 加载数据
boston = fetch_openml(name='boston', version=1, parser='auto')
X = boston.data.values if hasattr(boston.data, 'values') else boston.data
y = boston.target.values if hasattr(boston.target, 'values') else boston.target

# 划分和归一化
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 准备数据
m, n = X_train_scaled.shape
X = np.concatenate([np.ones((m, 1)), X_train_scaled], axis=1)
w = np.zeros(n + 1)
learning_rate = 0.01

print('前三步梯度下降详细计算')
print('=' * 80)

for epoch in range(3):
    print(f'\n第 {epoch + 1} 步迭代:')
    print('-' * 80)
    
    # 计算预测值
    h = np.dot(X, w)
    
    # 计算损失
    loss = np.sum((h - y_train) ** 2) / (2 * m)
    print(f'损失函数 J(w) = {loss:.6f}')
    
    # 计算误差
    diff = h - y_train
    
    # 显示前5个参数的更新
    print(f'\n参数更新 (前5个参数):')
    print(f'{"参数":<12} {"更新前":<15} {"梯度":<15} {"学习率×梯度":<15} {"更新后":<15}')
    print('-' * 80)
    
    for j in range(min(5, len(w))):
        gradient = np.sum(diff * X[:, j]) / m
        w_old = w[j]
        w_new = w_old - learning_rate * gradient
        
        param_name = f'w[{j}]' if j > 0 else 'w[0](偏置)'
        print(f'{param_name:<12} {w_old:<15.6f} {gradient:<15.6f} {learning_rate * gradient:<15.6f} {w_new:<15.6f}')
    
    # 更新所有参数
    for j in range(len(w)):
        gradient = np.sum(diff * X[:, j]) / m
        w[j] = w[j] - learning_rate * gradient
    
    print(f'\n... (共 {len(w)} 个参数)')
