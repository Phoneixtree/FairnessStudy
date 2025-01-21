import pandas as pd
import collections

df1=pd.read_csv("../data/train_data_OULAD.csv",header=0)

result=collections.defaultdict(list)

for index,value in enumerate(df1["region"]):
    result[value].append(index)

print(result)
