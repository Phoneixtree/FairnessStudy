import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
df = pd.read_csv('../data/final_comparison.csv')  # 替换为你的 CSV 文件路径

grouped = df.groupby('actual').agg({'calibrated_prediction': 'mean', 'origin_prediction': 'mean'}).reset_index()

# 获取横坐标和纵坐标
x = grouped['actual']  # 第三列作为横坐标
y1 = grouped['calibrated_prediction']  # 第一列的均值作为纵坐标1
y2 = grouped['origin_prediction']  # 第二列的均值作为纵坐标2

# 创建图形
plt.plot(x, y1, label='after calibration')
plt.plot(x, y2, label='before calibration')

# 添加标题和标签
plt.title('Sufficiency for OULAD')
plt.xlabel('actual')
plt.ylabel('predicted')

# 显示图例
plt.legend()

# 显示图形
plt.show()
