import pandas as pd
import matplotlib.pyplot as plt

# 读取第一个CSV文件的数据
df1 = pd.read_csv('./fair_trail_OULAD.csv')
individual1 = df1.iloc[:, 0]
accuracy1 = df1.iloc[:, 2]

# 读取第二个CSV文件的数据
df2 = pd.read_csv('./fair_trail_OULAD1.csv')
individual2 = df2.iloc[:, 0]
accuracy2 = df2.iloc[:, 2]

# 选择accuracy1和accuracy2中较小的作为横坐标的最大值
max_accuracy = min(accuracy1.max(), accuracy2.max())

plt.figure(figsize=(10, 6))

# 绘制第一个CSV文件的曲线
plt.plot(accuracy1[accuracy1 <= max_accuracy], individual1[accuracy1 <= max_accuracy], label='Multicalibration', color='purple', marker='o')

# 过滤第二个CSV文件的点
mask = individual2 <= 0.008
filtered_accuracy2 = accuracy2[mask]
filtered_individual2 = individual2[mask]

# 只绘制小于等于max_accuracy的点
filtered_accuracy2 = filtered_accuracy2[filtered_accuracy2 <= max_accuracy]
filtered_individual2 = filtered_individual2[filtered_accuracy2.index]

# 绘制第二个CSV文件的曲线
plt.plot(filtered_accuracy2, filtered_individual2, label='Fairness-bounary Multicalibration', color='blue', marker='o')

# 绘制公平损失阈值虚线
plt.axhline(y=0.008, color='purple', linestyle='--', linewidth=1)

# 添加标签
plt.ylabel('Fairness Loss')
plt.xlabel('Accuracy')

# 添加图例
plt.legend(loc='upper left')

plt.show()
