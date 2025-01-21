import matplotlib.pyplot as plt
import numpy as np

# 提供的数据
tpr1 = [0.13861056125606896, 0.13374428824675264, 0.08308526884634504, 0.1307242179120163, 0.04834605597964376]
tpr2 = [0.17634631948146572, 0.1342480870172659, 0.15017876706547944, 0.15423224788398326, 0.097299029309207425]
fpr1 = [0.14032488180379163, 0.13997565099179102, 0.13932611460430255, 0.14270151527159944, 0.14285714285714285]
fpr2 = [0.12593304590435837, 0.13196250038385874, 0.1303067129805239, 0.13916541920976102, 0.13595880770900162]

# 设置柱形图的宽度
bar_width = 0.35

# 计算柱形图的位置
index = np.arange(len(tpr1))

# 绘制 tpr 的柱形图
plt.figure(figsize=(10, 6))
bar1 = plt.bar(index, tpr1, bar_width, label='before calibration', alpha=0.8, color='b')
bar2 = plt.bar(index + bar_width, tpr2, bar_width, label='after calibration', alpha=0.8, color='g')

# 添加标签、标题和图例
plt.xlabel('Samples')
plt.ylabel('TPR')
plt.title('Comparison of TPR1 and TPR2')
plt.xticks(index + bar_width / 2, ['group1', 'group3', 'group4', 'group7', 'group9'])
plt.legend()

# 显示图形
plt.tight_layout()
plt.show()

# 绘制 fpr 的柱形图
plt.figure(figsize=(10, 6))
bar1 = plt.bar(index, fpr1, bar_width, label='before calibration', alpha=0.8, color='b')
bar2 = plt.bar(index + bar_width, fpr2, bar_width, label='after calibration', alpha=0.8, color='g')

# 添加标签、标题和图例
plt.xlabel('Samples')
plt.ylabel('FPR')
plt.title('Comparison of FPR1 and FPR2')
plt.xticks(index + bar_width / 2, ['group1', 'group3', 'group4', 'group7', 'group9'])
plt.legend()

# 显示图形
plt.tight_layout()
plt.show()
