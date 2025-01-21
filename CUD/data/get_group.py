import pandas as pd

df1=pd.read_csv("../data/train_data_CUD.csv",header=0)

result=[]
for index,value in enumerate(df1["Admission_Channel"]):
    if value==7:
        result.append(index)

print(result)
