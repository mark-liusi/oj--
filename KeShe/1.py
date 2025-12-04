"""
波士顿房价预测项目
使用 LinearRegression (正规方程) 和 SGDRegressor (随机梯度下降) 进行预测
"""

import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings('ignore')
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)

# 提取特征和目标变量
X = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
y = raw_df.values[1::2, 2]

# 创建一个类似 load_boston() 返回的对象来保持兼容性
class Boston:
    pass

boston = Boston()
boston.data = X
boston.target = y
boston.feature_names = np.array([
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS',
    'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
])
boston.DESCR = """Boston Housing Dataset
Number of Instances: 506
Number of Attributes: 13
Attribute Information:
    CRIM     - per capita crime rate by town
    ZN       - proportion of residential land zoned for lots over 25,000 sq.ft.
    INDUS    - proportion of non-retail business acres per town
    CHAS     - Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
    NOX      - nitric oxides concentration (parts per 10 million)
    RM       - average number of rooms per dwelling
    AGE      - proportion of owner-occupied units built prior to 1940
    DIS      - weighted distances to five Boston employment centres
    RAD      - index of accessibility to radial highways
    TAX      - full-value property-tax rate per $10,000
    PTRATIO  - pupil-teacher ratio by town
    B        - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
    LSTAT    - % lower status of the population
Target Variable:
    MEDV     - median value of homes in $1000s"""

# a) 查看数据集的描述、特征名、标签名、数据样本量等信息
print("\n【数据集描述】")
print(boston.DESCR)

print("\n【特征名称】")
print("特征数:", len(boston.feature_names))
print("特征列表:", boston.feature_names)

print("\n【标签名称】")
print("标签: MEDV (中位房价)")

print("\n【数据样本量】")
print("总样本数:", boston.data.shape[0])
print("特征数:", boston.data.shape[1])

# b) 获取样本的特征数据和标签数据
X = boston.data  # 特征数据 (506, 13)
y = boston.target  # 标签数据 (506,)

print("\n【特征数据形状】", X.shape)
print("【标签数据形状】", y.shape)
print("\n前5个样本的特征数据:\n", X[:5])
print("\n前5个样本的标签数据:\n", y[:5])

# ============================================================================
# (2) 划分数据（分成训练集和测试集）
# ============================================================================
print("\n" + "=" * 80)
print("(2) 数据划分")
print("=" * 80)

# 使用 train_test_split 划分数据，测试集占 20%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\n训练集样本数:", X_train.shape[0])
print("测试集样本数:", X_test.shape[0])
print("训练集特征维度:", X_train.shape[1])
print("测试集特征维度:", X_test.shape[1])

# ============================================================================
# (3) 数据归一化
# ============================================================================
print("\n" + "=" * 80)
print("(3) 数据归一化")
print("=" * 80)

# 使用 StandardScaler 进行标准化（使用训练集计算均值和标准差）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n【训练集归一化前】")
print("第1个特征的均值:", X_train[:, 0].mean())
print("第1个特征的标准差:", X_train[:, 0].std())

print("\n【训练集归一化后】")
print("第1个特征的均值:", X_train_scaled[:, 0].mean())
print("第1个特征的标准差:", X_train_scaled[:, 0].std())

# ============================================================================
# (4) 训练模型
# ============================================================================
print("\n" + "=" * 80)
print("(4) 模型训练")
print("=" * 80)

# a) 使用正规方程 (LinearRegression) 建模
print("\n【模型 A: LinearRegression (正规方程)】")
model_lr = LinearRegression()
model_lr.fit(X_train_scaled, y_train)
print("✓ 模型训练完成")
print("模型系数 (前5个):", model_lr.coef_[:5])
print("模型截距:", model_lr.intercept_)

# b) 使用随机梯度下降 (SGDRegressor) 建模
print("\n【模型 B: SGDRegressor (随机梯度下降)】")
model_sgd = SGDRegressor(
    loss='squared_error',
    max_iter=5000,
    tol=1e-4,
    random_state=42,
    learning_rate='invscaling',
    eta0=0.001,
    power_t=0.25,
    penalty='l2',
    alpha=0.001,
    validation_fraction=0.1,
    n_iter_no_change=50,
    verbose=0,
    early_stopping=True
)

# 手动进行小批次训练以增加稳定性
print("正在进行随机梯度下降训练...")
batch_size = 32
n_batches = (X_train_scaled.shape[0] + batch_size - 1) // batch_size

for i in range(200):  # 进行多个 epoch
    indices = np.random.permutation(X_train_scaled.shape[0])
    X_shuffled = X_train_scaled[indices]
    y_shuffled = y_train[indices]
    
    for j in range(n_batches):
        start_idx = j * batch_size
        end_idx = min((j + 1) * batch_size, X_train_scaled.shape[0])
        X_batch = X_shuffled[start_idx:end_idx]
        y_batch = y_shuffled[start_idx:end_idx]
        
        if i == 0 and j == 0:
            model_sgd.fit(X_batch, y_batch)
        else:
            model_sgd.partial_fit(X_batch, y_batch)

print("✓ 模型训练完成")
print("模型系数 (前5个):", model_sgd.coef_[:5])
print("模型截距:", model_sgd.intercept_)

# ============================================================================
# (5) 模型评估
# ============================================================================
print("\n" + "=" * 80)
print("(5) 模型评估")
print("=" * 80)

# 进行预测
y_pred_lr = model_lr.predict(X_test_scaled)
y_pred_sgd = model_sgd.predict(X_test_scaled)

# 计算评估指标
print("\n【模型 A: LinearRegression (正规方程)】")
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)

print(f"MSE (均方误差):  {mse_lr:.6f}")
print(f"RMSE (均方根误差): {rmse_lr:.6f}")
print(f"R² 值:         {r2_lr:.6f}")

print("\n【模型 B: SGDRegressor (随机梯度下降)】")
mse_sgd = mean_squared_error(y_test, y_pred_sgd)
r2_sgd = r2_score(y_test, y_pred_sgd)
rmse_sgd = np.sqrt(mse_sgd)

print(f"MSE (均方误差):  {mse_sgd:.6f}")
print(f"RMSE (均方根误差): {rmse_sgd:.6f}")
print(f"R² 值:         {r2_sgd:.6f}")

# 模型对比
print("\n" + "=" * 80)
print("模型对比总结")
print("=" * 80)
print(f"\n{'模型':<30} {'MSE':<15} {'R² 值':<15}")
print("-" * 60)
print(f"{'LinearRegression (正规方程)':<30} {mse_lr:<15.6f} {r2_lr:<15.6f}")
print(f"{'SGDRegressor (随机梯度下降)':<30} {mse_sgd:<15.6f} {r2_sgd:<15.6f}")

if mse_lr < mse_sgd:
    print(f"\n✓ LinearRegression 模型表现更好 (MSE 更小)")
else:
    print(f"\n✓ SGDRegressor 模型表现更好 (MSE 更小)")

# ============================================================================
# 额外分析：预测结果可视化
# ============================================================================
print("\n" + "=" * 80)
print("预测结果分析（前10个样本）")
print("=" * 80)

results_df = pd.DataFrame({
    '真实房价': y_test[:10],
    'LR预测': y_pred_lr[:10],
    'SGD预测': y_pred_sgd[:10],
    'LR误差': np.abs(y_test[:10] - y_pred_lr[:10]),
    'SGD误差': np.abs(y_test[:10] - y_pred_sgd[:10])
})

print("\n", results_df.to_string(index=False))
