import pandas as pd
import collections

def get_grade(gpa):
    if gpa >= 0.8:
        return "Distinction"
    elif gpa >= 0.6:
        return "Pass"
    else:
        return "Fail"

df = pd.read_csv('comparison0.csv')
calibration_f=df['calibration_prediction']
o=df['origin']
t=df['calibration_true']
I=df['individual_f']

M=545
before=0
after=0

accu_f=len([i for i in range(M) if get_grade(calibration_f[i])==get_grade(t[i])])/M
accu_i=len([i for i in range(M) if get_grade(I[i])==get_grade(t[i])])/M
print("before individual calibration, the total accuracy is {:.3f}".format(accu_f))
print("after individual calibration, the total accuracy is {:.3f}".format(accu_i))

average_f=[]
average_i=[]

""" for I in range(M):
    average_f.append(abs(calibration_f[I]-t[I]))
    average_i.append(abs(i[I]-t[I])) """

""" print("before individual calibration, the absolute distance is {:.3f}".format(sum(average_f)))
print("after individual calibration, the absolute is {:.3f}".format(sum(average_i))) """

""" df2 = pd.read_csv('train_data.csv')
adm=df2['Admission_Channel']
count=collections.Counter(adm)
indexes=collections.defaultdict(list)

for key in count.keys():
    indexes[key]=[i for i in range(M) if adm[i]==key]

accuracy_f=[]
accuracy_i=[]

for key in indexes.keys():
    accuracy_f.append(len([i for i in indexes[key] if get_grade(calibration_f[i])==get_grade(t[i])])/len(indexes[key]))
    accuracy_i.append(len([i for i in indexes[key] if get_grade(I[i])==get_grade(t[i])])/len(indexes[key]))

print("before individual calibration, the group accuracy is {}".format(accuracy_f))
print("before individual calibration, the group accuracy is {}".format(accuracy_i))
 """