import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter

# 读取数据
df = pd.read_csv('./fair_trail_OULAD.csv')

# 提取数据
individual = df.iloc[:, 0]
accuracy = [sum(float(d))/3 for d in df.iloc[:, 2]]
#accuracy=df.iloc[:,2]

plt.figure(figsize=(10, 6))

# 遍历每一对连续点
for i in range(len(individual) - 1):
    # 检查当前点和下一个点的条件
    if individual[i] > 0.008 or individual[i + 1] > 0.008:
        # 绘制紫线
        plt.plot(accuracy[i:i + 2], individual[i:i + 2], color='purple', marker='o', alpha=0.7)
    else:
        # 绘制蓝线
        plt.plot(accuracy[i:i + 2], individual[i:i + 2], color='blue', marker='o', alpha=0.7)

# 绘制fairness loss=0.008的虚线
plt.axhline(y=0.008, color='purple', linestyle='--', linewidth=1)

# 添加标签
plt.ylabel('Fairness Loss')
plt.xlabel('Accuracy')



# 自定义图例项
custom_legend = [
    Line2D([0], [0], color='blue', marker='o', label='Fairness Loss<=0.008 (Remained)'),     # 蓝线
    Line2D([0], [0], color='purple', marker='o', label='Fairness Loss>0.008 (Discarded))'), # 紫线
    Line2D([0], [0], color='purple', linestyle='--', label='Fairness Loss = 0.008 boundary')  # 虚线
]

# 显示图例
plt.legend(handles=custom_legend, loc='upper left')

plt.show()
