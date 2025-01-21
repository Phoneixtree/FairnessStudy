import pandas as pd

# 读取 CSV 文件
df1 = pd.read_csv("prediction_OULAD.csv")
df2 = pd.read_csv("true_level_OULAD.csv")

# 将字符串映射到数字的函数
def str_to_number(str_val):
    str_map = {
        "Distinction": 0.9, 
        "Pass": 0.7, 
        "Fail": 0.5
    }
    return str_map.get(str_val, 0)

# 获取行数和列数
m = len(df1)
n = len(df1.columns)

Predicted = []
Actual = []

# 计算 Predicted 和 Actual
for i in range(m):
    Predicted.append(sum(str_to_number(d) for d in df1.iloc[i]) / n)  # 对 df1 的每行计算平均值
    Actual.append(str_to_number(df2.iloc[i, 0]))  # df2 中的每行只对应一个值

# 创建新的 DataFrame
output_df = pd.DataFrame({
    'Predicted': Predicted,
    'Actual': Actual
})

# 输出到新的 CSV 文件
output_df.to_csv('prediction_numerical1.csv', index=False)
