"""
SMO（Sequential Minimal Optimization）算法实现线性 SVM 分类器模板
基于 numpy 实现，不使用 scikit-learn
"""

import numpy as np
from typing import Tuple

class SimpleSMO:
    """
    简化版 SMO 算法实现线性 SVM
    
    优化目标：
    max: sum(alpha_i) - 0.5 * sum(alpha_i * alpha_j * y_i * y_j * K(x_i, x_j))
    约束条件：
    - 0 <= alpha_i <= C
    - sum(alpha_i * y_i) = 0
    
    其中 K(x_i, x_j) 是核函数，这里使用线性核 K(x_i, x_j) = x_i^T @ x_j
    """
    
    def __init__(self, C=1.0, max_iter=100, tol=1e-3, kernel='linear'):
        """
        初始化 SVM 分类器
        
        参数：
        - C: 惩罚系数，控制松弛变量的惩罚
        - max_iter: 最大迭代次数
        - tol: 容差，用于判断 KKT 条件
        - kernel: 核函数类型，这里只实现 'linear'
        """
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.kernel = kernel
        
        # 模型参数
        self.alpha = None  # 拉格朗日乘数
        self.b = 0  # 截距
        self.w = None  # 权重向量（用于线性核）
        self.support_vectors = None  # 支持向量的索引
        self.X = None  # 训练数据
        self.y = None  # 训练标签
        
    def _kernel(self, x1, x2):
        """计算核函数"""
        if self.kernel == 'linear':
            return np.dot(x1, x2.T)
        else:
            raise ValueError(f"Unsupported kernel: {self.kernel}")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SimpleSMO':
        """
        使用 SMO 算法训练 SVM 模型
        
        参数：
        - X: 训练数据，形状为 (n_samples, n_features)
        - y: 训练标签，形状为 (n_samples,)，值为 +1 或 -1
        
        返回：
        - self
        """
        m, n = X.shape
        self.X = X
        self.y = y
        
        # 初始化拉格朗日乘数
        self.alpha = np.zeros(m)
        
        # 计算核矩阵
        K = self._kernel(X, X)  # 形状为 (m, m)
        
        # 迭代优化
        for iteration in range(self.max_iter):
            alpha_pairs_changed = 0
            
            for i in range(m):
                # 计算第 i 个样本的预测值
                E_i = self._decision_function(X[i:i+1])[0] - y[i]
                
                # 检查 KKT 条件
                if (y[i] * E_i < -self.tol and self.alpha[i] < self.C) or \
                   (y[i] * E_i > self.tol and self.alpha[i] > 0):
                    
                    # 选择第二个变量
                    j = self._select_j(i, m)
                    
                    # 计算第 j 个样本的预测值
                    E_j = self._decision_function(X[j:j+1])[0] - y[j]
                    
                    # 保存旧的 alpha 值
                    alpha_i_old = self.alpha[i]
                    alpha_j_old = self.alpha[j]
                    
                    # 计算 alpha 的上下界
                    if y[i] != y[j]:
                        L = max(0, self.alpha[j] - self.alpha[i])
                        H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
                    else:
                        L = max(0, self.alpha[i] + self.alpha[j] - self.C)
                        H = min(self.C, self.alpha[i] + self.alpha[j])
                    
                    if L == H:
                        continue
                    
                    # 计算二阶导数
                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue
                    
                    # 更新 alpha_j
                    self.alpha[j] -= y[j] * (E_i - E_j) / eta
                    self.alpha[j] = np.clip(self.alpha[j], L, H)
                    
                    if abs(self.alpha[j] - alpha_j_old) < 1e-5:
                        continue
                    
                    # 更新 alpha_i
                    self.alpha[i] += y[i] * y[j] * (alpha_j_old - self.alpha[j])
                    
                    # 更新截距 b
                    b1 = self.b - E_i - y[i] * (self.alpha[i] - alpha_i_old) * K[i, i] - \
                         y[j] * (self.alpha[j] - alpha_j_old) * K[i, j]
                    b2 = self.b - E_j - y[i] * (self.alpha[i] - alpha_i_old) * K[i, j] - \
                         y[j] * (self.alpha[j] - alpha_j_old) * K[j, j]
                    
                    if 0 < self.alpha[i] < self.C:
                        self.b = b1
                    elif 0 < self.alpha[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2
                    
                    alpha_pairs_changed += 1
            
            if alpha_pairs_changed == 0:
                print(f"收敛于第 {iteration} 次迭代")
                break
        
        # 计算权重向量
        self.w = np.dot(self.alpha * self.y, X)
        
        # 找出支持向量（alpha > 0 的样本）
        self.support_vectors = np.where(self.alpha > 1e-5)[0]
        
        return self
    
    def _select_j(self, i: int, m: int) -> int:
        """选择第二个变量（简单方法：随机选择非 i 的样本）"""
        j = i
        while j == i:
            j = np.random.randint(0, m)
        return j
    
    def _decision_function(self, X: np.ndarray) -> np.ndarray:
        """计算决策函数值 f(x) = w^T @ x + b"""
        return np.dot(X, self.w) + self.b
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测样本的类别"""
        decision = self._decision_function(X)
        return np.sign(decision)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """计算分类准确率"""
        predictions = self.predict(X)
        return np.mean(predictions == y)


# 使用示例
if __name__ == '__main__':
    from sklearn.datasets import load_iris
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    
    # 加载数据
    iris = load_iris()
    X = iris.data[:, 2:4]  # 选择花瓣长度和花瓣宽度
    y = iris.target
    
    # 二分类：选择类别 1 和 2
    mask = (y == 1) | (y == 2)
    X = X[mask]
    y = y[mask]
    y = np.where(y == 1, -1, 1)  # 转换标签为 -1 和 1
    
    # 划分数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # 训练模型
    svm = SimpleSMO(C=1.0, max_iter=100)
    svm.fit(X_train, y_train)
    
    # 评估
    print(f"训练集准确率: {svm.score(X_train, y_train):.4f}")
    print(f"测试集准确率: {svm.score(X_test, y_test):.4f}")
