import itertools
import math
import json
import numpy as np
import pandas as pd
import random
from FairnessLoss import FairnessLoss

class MCboost():
    def __init__(self):
        with open('./BoostSetting_OULAD.json', 'r') as json_file:
            info = json.load(json_file)
        #p=Preprocess()
        preprocessed=pd.read_csv(info["prediction_numerical"])
        self.f=preprocessed["predicted"] #predict
        self.origin=self.f.copy()
        self.t=preprocessed["actual"] #true
        #self.loss=p.loss #initial loss
        self.levels=info["level_length"]
        self.n=info["data_size"]
        self.indexes=np.arange(self.n)
        self.m=info["fetch_size"] #each fetch
        self.alpha=(np.max(self.f)-np.min(self.f))/self.levels
        self.lambda_discretization=[(i+1/2)*self.alpha+np.min(self.f) for i in range(self.levels-1)]
        self.I_min=self.n*math.log(self.n)/self.m #iteration

        self.fairness_border=info["fairness_border"]
        self.comparison_trail=info["comparison_trail"]
        self.individual_loss=[]
        self.group_loss=[]
        self.accuracy=[]
        self.overflow=0
        self.run()

    def discretization(self,S):
        S_v={}
        for l in self.lambda_discretization:
            s_lambda=[index for index,f in enumerate(self.f) if l-self.alpha/2<=f<l+self.alpha/2 and index in S]
            S_v[l]=s_lambda
        return S_v

    def subset(self,set):
        tmp=[]
        for n in range(1,len(set)+1):
            tmp.extend(itertools.combinations(set,n))
        return tmp

    def guess_and_check(self,S,v,omega):
        p_s=sum([self.t[s] for s in S])/len(S)
        if math.fabs(p_s-v)<2*omega:
            return [True,0]
        else:
            return [False,p_s-omega]

    def fairness_check(self):
        df = pd.DataFrame()
        df['calibrated_prediction']=self.f
        df['origin_prediction']=self.origin
        df['actual']=self.t
        df.to_csv(self.comparison_trail, index=False, header=['calibrated_prediction', 'origin_prediction','actual'])
        f=FairnessLoss()
        individual_loss=f.individual_loss(f.cali_prediction)
        group_loss=f.group_loss(f.cali_prediction)
        accuracy=f.accuracy()
        self.individual_loss.append(individual_loss)
        self.group_loss.append(group_loss)
        self.accuracy.append(accuracy)
        if individual_loss>self.fairness_border:
            return False
        else:
            return True

    def overflow_check(self):
        if self.overflow>5:
            return False
        return True

    def boost(self):
        print("Running MCboost...........................")
        no_update=0
        while no_update<self.I_min and self.overflow_check():
            no_update+=1
            C=self.subset(np.random.choice(self.indexes, self.m, replace=False))
            tmp_f=self.f.copy()
            for S in C:
                S_v=self.discretization(S)
                for key in self.lambda_discretization:
                    S_v_current=S_v[key]
                    """ if len(S_v_current)<(self.alpha**2)*len(S):
                        continue """
                    if len(S_v_current)==0:
                        continue
                    v_average=sum([self.f[s] for s in S_v_current])/len(S_v_current)
                    r=self.guess_and_check(S_v_current,v_average,self.alpha/4)
                    if not r[0]:
                        #no_update=0
                        for s in S_v_current:
                            self.f[s]+=r[1]-v_average
            if not self.fairness_check():
                self.f=tmp_f
                self.alpha+=self.alpha if random.random()>0.5 else 0
                self.overflow+=1
            else:
                self.overflow=0
                continue

    def run(self):
        self.boost()