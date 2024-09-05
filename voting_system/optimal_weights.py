import numpy as np
from scipy.optimize import minimize

class Optimal:
    def __init__(self):
        self.accuracy = np.array([0.31, 0.34, 0.18, 0.37, 0.31, 0.28, 0.37])#准确率数组
        self.initial_weights = np.ones(len(self.accuracy)) / len(self.accuracy)#初始权重猜测（均等分配）
        self.cons = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})#约束条件：权重和为1
        self.optimal_weights = self.min_optimal()

    def objective(self, weights):#定义目标函数（取负值以进行最小化）
        return -np.sum(weights * self.accuracy)
    
    def min_optimal(self):
        bounds = tuple((0, 1) for _ in range(len(self.accuracy)))#权重的界限：每个权重都应该在0和1之间
        result = minimize(self.objective, self.initial_weights, method='SLSQP', bounds=bounds, constraints=self.cons)  # 使用SLSQP方法进行优化
        return result.x