import pandas as pd

# 读取CSV文件
df = pd.read_csv('train_data1.csv')

# 删除名为'a'的列
df = df.drop(columns=['Admission_Channel'])

# 将结果保存为新的CSV文件
df.to_csv('train_data1.csv', index=False)
