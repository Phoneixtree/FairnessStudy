import numpy as np
import pandas as pd
import json
from ConfusionMatrix import ConfusionMatrix

def get_grade(gpa):
    if gpa>= 0.672:
        return "A"
    elif gpa>=0.466:
        return "B+"
    elif gpa>=0.283:
        return "B"
    elif gpa>=0.169:
        return "B-"
    elif gpa>=0.044:
        return "C"
    elif gpa>=0.026:
        return "D"
    else:
        return "F"
    
df1=pd.read_csv("../data/predicted_numerical.csv")
df2=pd.read_csv("../data/result_numerical.csv")

predicted1=[get_grade(p) for p in df1["predicted"]]
actual1=[get_grade(p) for p in df1["actual"]]

predicted2=[get_grade(p) for p in df2["predicted"]] 
actual2=[get_grade(p) for p in df2["actual"]]

cm1=ConfusionMatrix({"actual":actual1,"predicted":predicted1})
cm2=ConfusionMatrix({"actual":actual2,"predicted":predicted2})

with open("../Settings_CUD.json","r") as json_file:
    info = json.load(json_file)
groups=["group1","group3","group4","group7","group9"]

tpr1=[]
tpr2=[]
fpr1=[]
fpr2=[]

for g in groups:
    group=info[g]
    actual1_tmp=[a for i,a in enumerate(actual1) if i in group]
    predicted1_tmp=[p for i,p in enumerate(predicted1) if i in group]
    cm1_tmp=ConfusionMatrix({"actual":actual1_tmp,"predicted":predicted1_tmp})

    actual2_tmp=[a for i,a in enumerate(actual2) if i in group]
    predicted2_tmp=[p for i,p in enumerate(predicted2) if i in group]
    cm2_tmp=ConfusionMatrix({"actual":actual2_tmp,"predicted":predicted2_tmp})

    tpr1.append(cm1_tmp.true_positive_rate(cm1_tmp.matrix))
    fpr1.append(cm1_tmp.false_positive_rate(cm1_tmp.matrix))
    tpr2.append(cm2_tmp.true_positive_rate(cm2_tmp.matrix))
    fpr2.append(cm2_tmp.false_positive_rate(cm2_tmp.matrix))

df3=pd.DataFrame({"tpr1":tpr1,"tpr2":tpr2,"fpr1":fpr1,"fpr2":fpr2})
df3.to_csv("../data/measurement.csv",index=False,header=["tpr1","tpr2","fpr1","fpr2"])
