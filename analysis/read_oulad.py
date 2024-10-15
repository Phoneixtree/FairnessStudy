import pandas as pd
from collections import Counter

df1=pd.read_csv("../data/train_data_OULAD.csv")
df2=pd.read_csv("../data/studentAssessment.csv")

counter1=Counter(df1["final_result"])

id_score=zip(df2["id_student"],df2["score"])
id=df1["id_student"]
result=df1["final_result"]
id_result=zip(id,result)

dict2={item[0]:item[1] for item in id_score if item[1]!=""}

D_sum=0
D_max=0
D_min=100

P_sum=0
P_max=0
P_min=100

W_sum=0
W_max=0
W_min=100

F_sum=0
F_max=0
F_min=100

for pair in id_result:
    if pair[0] not in dict2.keys():
        continue
    if pair[1]=="Distinction":
        tmp=dict2[pair[0]]
        D_sum+=tmp
        D_max=max(D_max,tmp)
        D_min=min(D_min,tmp)
    if pair[1]=="Pass":
        tmp=dict2[pair[0]]
        P_sum+=tmp
        P_max=max(P_max,tmp)
        P_min=min(P_min,tmp)
    if pair[1]=="Withdrawn":
        tmp=dict2[pair[0]]
        W_sum+=tmp
        W_max=max(W_max,tmp)
        W_min=min(W_min,tmp)
    else:
        tmp=dict2[pair[0]]
        F_sum+=tmp
        F_max=max(F_max,tmp)
        F_min=min(F_min,tmp)

print(D_sum,D_max,D_min)
print(P_sum,P_max,P_min)
print(W_sum,W_max,W_min)
print(F_sum,F_max,F_min)