import pandas as pd
import math

inn=pd.read_csv("output.csv",header=0).values
out=[]

def trans(n):
    tmp=math.ceil(n)
    if tmp <= 1: #A
        return 'A'
    if tmp == 2: #B+
        return 'B+'
    if tmp == 3: #B
        return 'B'
    if tmp == 4: #B-
        return 'B-'
    if tmp == 5: #C
        return 'C'
    if tmp == 6: #D 
        return 'D'
    if tmp == 7: #F
        return 'F'
    
for line in inn:
    out.append([trans(i) for i in line])

df = pd.DataFrame(out, columns=['KNN', 'GBT', 'NB', 'SVC', 'RF', 'DT', 'MLR'])

# 将DataFrame保存到CSV文件
df.to_csv('output.csv', index=False)
