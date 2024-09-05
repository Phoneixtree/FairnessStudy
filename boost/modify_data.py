import pandas as pd

# 读取CSV文件
df1 = pd.read_csv('train_data.csv')
df2 = pd.read_csv("train_data2.csv")

# 删除名为'a'的列
#df = df.drop(columns=['Admission_Channel'])

a_column=df1["Admission_Channel"]
df2["Admission_Channel"]=a_column

# 将结果保存为新的CSV文件
df2.to_csv('train_data1.csv', index=False)
