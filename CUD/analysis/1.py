import pandas as pd
import matplotlib.pyplot as plt


# 读取 TPR 和 FPR 数据
tpr_file = "../data/tpr_trail_0.1.csv"  # 替换为实际文件路径
fpr_file = "../data/fpr_trail_0.1.csv"  # 替换为实际文件路径

tpr_data = pd.read_csv(tpr_file)  # TPR 数据，形状为 (n_iterations, 5)
fpr_data = pd.read_csv(fpr_file)  # FPR 数据，形状为 (n_iterations, 5)

def equalized_odds_loss(tpr1,tpr2,fpr1,fpr2):
    return (tpr1-tpr2)**2+(fpr1-fpr2)**2

length=len(tpr_data)
loss_trail=[]
groups=["1","3","4","7"]

for l in range(length):
    tmp_loss=0
    for i in range(3):
        for j in range(i+1,4):
            group1=groups[i]
            group2=groups[j]
            tmp_loss+=equalized_odds_loss(tpr_data[group1][l],tpr_data[group2][l],fpr_data[group1][l],fpr_data[group2][l])
    loss_trail.append(tmp_loss)
plt.plot(loss_trail)
plt.title("Loss Trail Over Iterations")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.grid(True)
plt.show()




