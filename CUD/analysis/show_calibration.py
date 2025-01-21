import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. 读取 CSV 文件
csv_file = '../data/result_numerical.csv'  # 替换为你的 CSV 文件名
data = pd.read_csv(csv_file, header=0)  # 如果没有列名，使用 header=None
data = data.round(4)

#sorted_actual=data["actual"].sort_values()

#def find_pstar(value):
#    return np.searchsorted(sorted_actual, value)/data.shape[0]

# 提取第一列和第二列数据
y = data.iloc[:, 0]  # 第一列数据
x = data.iloc[:, 1]  # 第二列数据

#for i in range(data.shape[0]):
#    x[i]=find_pstar(x[i])

# 2. 绘制图像
plt.figure(figsize=(8, 6))

# 绘制点
plt.plot(x, y, marker='o', linestyle='None', color='b', label='Data Points')

# 添加基准线 y = x
plt.plot([0, 1], [0, 1], linestyle='--', color='green', label='y = x')  # 基准线

# 添加两条平行线（上下平移一定距离）
margin = 0.05  # 平移距离（根据需求调整）
plt.plot([0, 1], [margin, 1 + margin], linestyle='--', color='red', label=f'y = x + {margin}')
plt.plot([0, 1], [-margin, 1 - margin], linestyle='--', color='orange', label=f'y = x - {margin}')

# 调整数据点到范围内
y = y.clip(lower=x - margin, upper=x + margin)

# 绘制调整后的点
plt.scatter(x, y, color='purple', label='Adjusted Points', alpha=0.7)

# 设置标题和轴标签
plt.title('Calibration Result')
plt.xlabel('actual value')
plt.ylabel('predicted value')

# 设置 x 和 y 轴的范围为 0 到 1
plt.xlim(0, 1)
plt.ylim(0, 1)

# 添加网格和图例
plt.grid(True)
plt.legend()
plt.show()
