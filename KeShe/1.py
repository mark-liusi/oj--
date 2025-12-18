import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings


warnings.filterwarnings('ignore')
data_url = "http://lib.stat.cmu.edu/datasets/boston"
try:
    # 波士顿数据集的原始格式需要特殊处理
    # 该数据集包含 506 个样本，每个样本 13 个特征 + 1 个标签
    # 在原始文件中，每条记录占两行
    # 前 11 个特征在第一行，后 2 个特征和标签在第二行
    raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)
    
    # 处理数据：将两行合并为一行
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]
    
    feature_names = np.array(['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'])
    
    X = data
    y = target
    
    # 转换为 DataFrame 便于查看
    df = pd.DataFrame(X, columns=feature_names)
    df['PRICE'] = y
except Exception as e:
    print(f"数据加载失败: {e}")
    print("请检查网络连接或数据源地址。")
    exit(1)

# ==================== (2) 划分数据集 ====================
# 划分训练集和测试集 (80% 训练集, 20% 测试集)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"训练集样本数: {X_train.shape[0]}")
print(f"测试集样本数: {X_test.shape[0]}")

# ==================== (3) 数据归一化 ====================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==================== (4) 训练模型 ====================
# (a) 使用正规方程方法 (LinearRegression)
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

# (b) 使用随机梯度下降方法 (SGDRegressor)
sgd_model = SGDRegressor(
    max_iter=10000,
    tol=1e-3,
    random_state=42,
    learning_rate='invscaling',
    eta0=0.01
)
sgd_model.fit(X_train_scaled, y_train)

# ==================== (5) 模型评估 ====================
y_train_pred_lr = lr_model.predict(X_train_scaled)
y_test_pred_lr = lr_model.predict(X_test_scaled)

train_mse_lr = mean_squared_error(y_train, y_train_pred_lr)
test_mse_lr = mean_squared_error(y_test, y_test_pred_lr)
train_r2_lr = r2_score(y_train, y_train_pred_lr)
test_r2_lr = r2_score(y_test, y_test_pred_lr)

print(f"训练集 MSE: {train_mse_lr:.4f}")
print(f"测试集 MSE: {test_mse_lr:.4f}")
print(f"训练集 R²: {train_r2_lr:.4f}")
print(f"测试集 R²: {test_r2_lr:.4f}")

# 评估随机梯度下降模型

y_train_pred_sgd = sgd_model.predict(X_train_scaled)
y_test_pred_sgd = sgd_model.predict(X_test_scaled)

train_mse_sgd = mean_squared_error(y_train, y_train_pred_sgd)
test_mse_sgd = mean_squared_error(y_test, y_test_pred_sgd)
train_r2_sgd = r2_score(y_train, y_train_pred_sgd)
test_r2_sgd = r2_score(y_test, y_test_pred_sgd)

print(f"训练集 MSE: {train_mse_sgd:.4f}")
print(f"测试集 MSE: {test_mse_sgd:.4f}")
print(f"训练集 R²: {train_r2_sgd:.4f}")
print(f"测试集 R²: {test_r2_sgd:.4f}")

# 模型对比
comparison_df = pd.DataFrame({
    '评价指标': ['训练集 MSE', '测试集 MSE', '训练集 R²', '测试集 R²'],
    '正规方程': [train_mse_lr, test_mse_lr, train_r2_lr, test_r2_lr],
    '随机梯度下降': [train_mse_sgd, test_mse_sgd, train_r2_sgd, test_r2_sgd]
})

print(comparison_df.to_string(index=False))
print(f"  - 正规方程测试集 MSE: {test_mse_lr:.4f}")
print(f"  - SGD 测试集 MSE: {test_mse_sgd:.4f}")
print(f"  - 正规方程测试集 R²: {test_r2_lr:.4f}")
print(f"  - SGD 测试集 R²: {test_r2_sgd:.4f}")

if test_mse_lr < test_mse_sgd:
    print(f"\n正规方程方法表现更好 (测试集 MSE 更低)")
else:
    print(f"\n随机梯度下降方法表现更好 (测试集 MSE 更低)")
