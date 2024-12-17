import pandas as pd
import collections

df = pd.read_csv('./train_data_CUD.csv')
level=df["Level"]

counter=collections.Counter(level)

for key in counter.keys():
    counter[key]=counter[key]/545

levels=["A","B+","B","B-","C","D","F"]
for l in levels:
    print(l,counter[l])
