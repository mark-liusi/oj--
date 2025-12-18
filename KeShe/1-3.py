import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. 准备数据
print("正在加载数据...")
try:
    boston = fetch_openml(name='boston', version=1, parser='auto')
    X = boston.data.values if hasattr(boston.data, 'values') else boston.data
    y = boston.target.values if hasattr(boston.target, 'values') else boston.target
except:
    X = np.random.rand(506, 13)
    y = np.random.rand(506)

# 2. 中文特征名称映射
feature_map = [
    '人均犯罪率 (CRIM)', '住宅用地比例 (ZN)', '商业用地比例 (INDUS)', 
    '查尔斯河 (CHAS)', '一氧化氮浓度 (NOX)', '平均房间数 (RM)', 
    '老房比例 (AGE)', '就业中心距离 (DIS)', '高速公路便利 (RAD)', 
    '财产税率 (TAX)', '师生比例 (PTRATIO)', '黑人比例 (B)', 
    '低收入比 (LSTAT)'
]
# 添加截距
feature_names_cn = ['截距 (Bias)'] + feature_map

# 3. 预处理
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

m, n = X_train_scaled.shape
X_b = np.concatenate([np.ones((m, 1)), X_train_scaled], axis=1) # 添加偏置列
w = np.zeros(n + 1)
learning_rate = 0.01
epochs = 3

# 4. 计算并记录
excel_data = []

print("正在计算...")
for epoch in range(epochs):
    h = np.dot(X_b, w)
    loss = np.sum((h - y_train) ** 2) / (2 * m)
    diff = h - y_train
    
    # 保存当前权重用于记录
    current_w_copy = w.copy()
    
    # 先计算所有梯度并记录
    for j in range(len(w)):
        gradient = np.sum(diff * X_b[:, j]) / m
        w_old = current_w_copy[j]
        update = learning_rate * gradient
        w_new = w_old - update
        
        excel_data.append({
            '迭代轮数': epoch + 1,
            '当前损失': loss,
            '参数名称': feature_names_cn[j],
            '更新前权重': w_old,
            '梯度': gradient,
            '学习率': learning_rate,
            '更新量': update,
            '更新后权重': w_new
        })
    
    # 统一更新所有参数（批量梯度下降）
    for j in range(len(w)):
        gradient = np.sum(diff * X_b[:, j]) / m
        w[j] = w[j] - learning_rate * gradient

# 5. 保存为 CSV（不需要额外库）
df = pd.DataFrame(excel_data)
output_file = '波士顿房价_梯度下降_前3步计算详情.csv'
df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"成功！文件已保存为: {output_file}")
print(f"\n前10行数据预览:")
print(df.head(10))