from MCboost import MCboost
from Individual import Individual
import pandas as pd
import time

def main():
    start=time.perf_counter()
    m=MCboost()
    end=time.perf_counter()
    f=[round(i) for i in m.f]
    #o=[round(i) for i in m.origin]

    df = pd.DataFrame()
    df['calibration_prediction']=m.f
    df['origin']=m.origin
    df['calibration_true']=m.t
    df.to_csv("comparison.csv", index=False, header=['calibration_prediction', 'origin', 'calibration_true'])
    i=Individual({'attri':['Admission_Channel'],'f':m.f,'t':m.t,'levels':m.levels,'n':m.n})
    df.insert(loc=len(df.columns), column='individual_f', value=i.f)
    df.to_csv("comparison.csv", index=False,header='individual_f')
    print('running time:{:.3f} in seconds'.format(end-start))

if __name__=="__main__":
    main()