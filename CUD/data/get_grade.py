import pandas as pd
import collections

df=pd.read_csv("./predicted_numerical.csv")

predicted=df["predicted"]
actual=df["actual"]

def get_grade(gpa):
    if gpa>=90:
        return "A"
    elif gpa>=80:
        return "B+"
    elif gpa>=70:
        return "B"
    elif gpa>=60:
        return "B-"
    elif gpa>=50:
        return "C"
    elif gpa>=40:
        return "D"
    else: