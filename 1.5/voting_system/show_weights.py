import numpy as np
import matplotlib.pyplot as plt

size = 7
x = np.array(['KNN', 'GBT', 'NB', 'SVC', 'RF', 'DT', 'MLR'])
a = np.array([0.9996189179198715, 1.0300618461962399, 0.8777608044871946, 1.0614318996654455, 0.9996189179198715, 0.970075714145932, 1.0614318996654455])
b = np.array([1, 1, 1, 1, 1, 1, 1])
c = 7 * np.array([0.13709998, 0.15023836, 0.08733484, 0.09536944, 0.17764675, 0.21932494, 0.1329857])

plt.figure(figsize=(10, 6))

bar_width = 0.25
index = np.arange(size)
plt.bar(index, a, bar_width, label='Exponentiation&Homogenization', color='g', alpha=0.5)
plt.bar(index + bar_width, b, bar_width, label='Uniform', color='y', alpha=0.5)
plt.bar(index + 2 * bar_width, c, bar_width, label='Gradient Descent', color='r', alpha=0.5)

# 添加标签、标题和图例
plt.xlabel('Methods')
plt.ylabel('Weights')
plt.title('Accuracy:Green<Yellow<Red')
plt.xticks(index + bar_width, x)
plt.legend()

# 显示图形
plt.tight_layout()
plt.show()