import pandas as pd
import matplotlib.pyplot as plt
import collections

#df2=pd.read_csv('../data/train_data_CUD.csv')
#df2=df2['Admission_Channel']
#print(collections.Counter(df2))

# 读取CSV文件
df = pd.read_csv('../data/tpr_trail_0.1.csv')

# 提取前五列
df_first_5_columns = df.iloc[:, :5]

# 存储每个索引的最大差值及其对应的列对
max_diff_indices = []
max_diff_values = []

# 对每个索引计算差值
for index, row in df_first_5_columns.iterrows():
    # 计算每两个值之间的差
    diffs = []
    for i in range(len(row)):
        for j in range(i+1, len(row)):
            diff = abs(row[i] - row[j])
            diffs.append((diff, (df_first_5_columns.columns[i], df_first_5_columns.columns[j])))
    
    # 找到最大差值
    max_diff, max_pair = max(diffs, key=lambda x: x[0])
    max_diff_indices.append(index)
    max_diff_values.append(max_diff)

# 绘制五条曲线
plt.figure(figsize=(10, 6))
for column in df_first_5_columns.columns:
    plt.plot(df_first_5_columns[column], label=column)

# 绘制最大差值线
plt.plot(max_diff_indices, max_diff_values, label='Max Difference', color='red', linestyle='--')

# 添加标题和标签
plt.title('Admission Channels')
plt.xlabel('Iterations')
plt.ylabel('TPR')
plt.legend()

# 显示图表
plt.show()

# 输出每个索引的最大差值及其对应的列对
for idx, diff, pair in zip(max_diff_indices, max_diff_values, max_diff_indices):
    print(f"Index {idx}: Max Difference = {diff} between {pair}")
