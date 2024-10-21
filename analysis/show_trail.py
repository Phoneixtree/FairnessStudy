import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv('./fair_trail_OULAD.csv')

# 过滤掉 y > 3 的数据点
individual = df.iloc[:, 0]
accuracy = df.iloc[:, 2]

# 设置画布大小
plt.figure(figsize=(10, 6))

# 绘制 y <= 3 的蓝色线和点
plt.plot(accuracy, individual, color='blue', marker='o', linestyle='-')

# 添加标签和标题
plt.ylabel('Fairness Loss')
plt.xlabel('Accuracy')

# 显示图例
plt.legend()

# 显示图形
plt.show()
