import pandas as pd

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

df = pd.read_csv('comparison0.csv')
calibration_f=df['calibration_prediction']
o=df['origin']
t=df['calibration_true']
i=df['individual_f']

M=545
before=0
after=0

average_f=[]
average_i=[]

for I in range(M):
    average_f.append(abs(calibration_f[I]-t[I]))
    average_i.append(abs(i[I]-t[I]))

print("before individual calibration, the absolute distance is {:.3f}".format(sum(average_f)))
print("after individual calibration, the absolute is {:.3f}".format(sum(average_i)))




""" for l in predictions:
    tmp0=get_grade(l[0])
    tmp1=get_grade(l[1])
    tmp2=get_grade(l[2])
    if tmp0==tmp1:
        after+=1
    if tmp2==tmp1:
        before+=1
 """
""" print("before calibration, the accuracy is {:.3f}".format(before/M))
print("after calibration, the accuracy is {:.3f}".format(after/M))
 """
