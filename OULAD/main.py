import sys
import pandas as pd
import time
import json

sys.path.append('./boost')
sys.path.append('./preprocess')
sys.path.append('./fairness')
sys.path.append('./matrix')

from MCboost import MCboost
from Preprocess import Preprocess

def main():
    with open("./Settings_OULAD.json","r") as json_file:
            info = json.load(json_file)

    start=time.perf_counter()
    preprocess=Preprocess()
    mcboost=MCboost()
    end=time.perf_counter()
    #f=[round(i) for i in m.f]
    #o=[round(i) for i in m.origin]

    df_tpr=pd.DataFrame()
    df_fpr=pd.DataFrame()
    """df['calibrated_prediction']=mcboost.predicted
    df['origin_prediction']=mcboost.predicted_origin
    df['actual']=mcboost.actual
    df.to_csv("../data/final_comparison.csv", index=False, header=['calibrated_prediction', 'origin_prediction', 'actual'])
    """

    for key in mcboost.tpr_trail:
        #df[key+"_TP_rate"]=mcboost.tp_trail[key]
        df_tpr[key]=mcboost.tpr_trail[key]
        df_fpr[key]=mcboost.fpr_trail[key]
    #df["accuracy"]=mcboost.accuracy_trail
    df_tpr.to_csv("./data/tpr_trail_"+str(info["alpha"])+".csv", index=False, header=df_tpr.keys())
    df_fpr.to_csv("./data/fpr_trail_"+str(info["alpha"])+".csv", index=False, header=df_fpr.keys())

    df2=pd.DataFrame()
    df2["predicted"]=mcboost.predicted
    df2["actual"]=mcboost.actual
    df2.to_csv("./data/result_numerical.csv",index=False,header=["predicted","actual"])
    """     df2=pd.DataFrame()
    df2["individual"]=m.individual_loss
    df2["group"]=m.group_loss
    df2["accuracy"]=m.accuracy
    df2.to_csv("fair_trail_OULAD.csv",index=False,header=["individual","group","accuracy"]) """

    print('running time:{:.3f} in seconds'.format(end-start))

if __name__=="__main__":
    main()