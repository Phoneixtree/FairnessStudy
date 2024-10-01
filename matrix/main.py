import pandas as pd
from ConfusionMatrix import ConfusionMatrix

def get_grade(gpa):
    if gpa >= 3.7:
        return "A"
    elif gpa >= 3.3:
        return "B+"
    elif gpa >= 3.0:
        return "B"
    elif gpa >= 2.7:
        return "B-"
    elif gpa >= 1.7:
        return "C"
    elif gpa >= 1.0:
        return "D"
    else:
        return "F"

def main():
    df = pd.read_csv('../boost/comparison.csv')
    calibration_f=df['calibration_prediction']
    o=df['origin']
    t=df['calibration_true']
    I=df['individual_f']
    info={'pred':[get_grade(i) for i in o],'true':[get_grade(i) for i in t]}
    conf=ConfusionMatrix(info)

if __name__=="__main__":
    main()