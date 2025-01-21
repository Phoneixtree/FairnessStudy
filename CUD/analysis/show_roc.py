import pandas as pd
import matplotlib.pyplot as plt

# 读取 TPR 和 FPR 数据
tpr_file = "tpr_trail_0.1.csv"  # 替换为实际文件路径
fpr_file = "fpr_trail_0.1.csv"  # 替换为实际文件路径

tpr_data = pd.read_csv(tpr_file)  # TPR 数据，形状为 (n_iterations, 5)
fpr_data = pd.read_csv(fpr_file)  # FPR 数据，形状为 (n_iterations, 5)

# 检查数据
assert tpr_data.shape == fpr_data.shape, "TPR and FPR files must have the same shape."
groups = tpr_data.columns  # 获取 group 名称

# 绘制 ROC 曲线
plt.figure(figsize=(10, 8))
for group in groups:
    plt.plot(fpr_data[group], tpr_data[group], label=f"Group {group}")

# 添加图例、标签和标题
plt.title("ROC Curves for 5 Groups")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.legend()
plt.grid(True)
plt.show()
