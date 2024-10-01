from MCboost import MCboost
from Individual import Individual
from Preprocess import Preprocess
import pandas as pd
import time

def main():
    start=time.perf_counter()
    p=Preprocess()
    m=MCboost()
    end=time.perf_counter()
    #f=[round(i) for i in m.f]
    #o=[round(i) for i in m.origin]

    df = pd.DataFrame()
    df['calibrated_prediction']=m.f
    df['origin_prediction']=m.origin
    df['actual']=m.t
    df.to_csv("../data/final_comparison.csv", index=False, header=['calibrated_prediction', 'origin_prediction', 'actual'])

    """ i=Individual({'attri':['Admission_Channel'],'f':m.f,'t':m.t,'levels':m.levels,'n':m.n})
    df.insert(loc=len(df.columns), column='individual_f', value=i.f)
    df.to_csv("comparison.csv", index=False,header='individual_f') """
    
    """     df2=pd.DataFrame()
    df2["individual"]=m.individual_loss
    df2["group"]=m.group_loss
    df2["accuracy"]=m.accuracy
    df2.to_csv("fair_trail2.csv",index=False,header=["individual","group","accuracy"]) """

    print('running time:{:.3f} in seconds'.format(end-start))

if __name__=="__main__":
    main()