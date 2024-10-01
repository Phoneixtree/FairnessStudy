import numpy as np
import pandas as pd

predictions = pd.read_csv('predict_outcome.csv', header=0).values#预测结果
true_labels = pd.read_csv('True_level.csv', header=0).values.flatten()#真实成绩

grade_encoding = {#定义成绩等级的编码
    'A': [1, 0, 0, 0, 0, 0, 0],
    'B+': [0, 1, 0, 0, 0, 0, 0],
    'B': [0, 0, 1, 0, 0, 0, 0],
    'B-': [0, 0, 0, 1, 0, 0, 0],
    'C': [0, 0, 0, 0, 1, 0, 0],
    'D': [0, 0, 0, 0, 0, 1, 0],
    'F': [0, 0, 0, 0, 0, 0, 1]
}
predictions_encoded = np.array([np.array([grade_encoding[grade] for grade in row]) for row in predictions])
true_labels_encoded = np.array([grade_encoding[grade] for grade in true_labels])

def loss_function(weights, predictions, true_labels):#损失函数,这里使用均方误差(MSE)
    weighted_predictions = np.dot(predictions, weights)
    return np.mean(np.sum((weighted_predictions - true_labels) ** 2, axis=1))

def compute_gradient(weights, predictions, true_labels):#计算梯度
    num_weights = len(weights)
    gradients = np.zeros(num_weights)
    for i in range(num_weights):#对每个权重微调
        epsilon = 1e-6
        weights_plus = np.copy(weights)
        weights_minus = np.copy(weights)
        weights_plus[i] += epsilon
        weights_minus[i] -= epsilon
        loss_plus = loss_function(weights_plus, predictions, true_labels)
        loss_minus = loss_function(weights_minus, predictions, true_labels)
        gradients[i] = (loss_plus - loss_minus) / (2 * epsilon)
    return gradients

weights = np.random.rand(7)#初始化权重
weights /= weights.sum()

learning_rate = 0.01#梯度下降参数
num_iterations = 1000

for iteration in range(num_iterations):#梯度下降循环
    gradients = compute_gradient(weights, predictions_encoded, true_labels_encoded)
    weights -= learning_rate * gradients
    weights = np.clip(weights, a_min=0, a_max=None)#确保权重不为负
    weights /= weights.sum()#重新归一化

    if iteration % 100 == 0:
        current_loss = loss_function(weights, predictions_encoded, true_labels_encoded)
        print(f"Iteration {iteration}, Loss: {current_loss}")

print("Optimal weights:", weights)