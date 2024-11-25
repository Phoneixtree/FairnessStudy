import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../data/final_comparison.csv')

x = df['actual']
y1 = df['calibrated_prediction']
y2 = df['origin_prediction']

# 创建图形
plt.plot(x, y1, label='after calibration')
plt.plot(x, y2, label='before calibration')

# 添加标题和标签
plt.title('')
plt.xlabel('X (col3)')
plt.ylabel('Y')

# 显示图例
plt.legend()

# 显示图形
plt.show()
