import pandas as pd
import collections

df = pd.read_csv('./train_data_OULAD.csv')
level=df["final_result"]

counter=collections.Counter(level)

for key in counter.keys():
    counter[key]=counter[key]/500

levels=["Distinction","Pass","Fail"]
for l in levels:
    print(l,counter[l])
