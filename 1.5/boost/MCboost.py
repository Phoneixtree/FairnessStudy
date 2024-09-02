from Preprocess import Preprocess
import itertools
import math
import numpy as np
import pandas as pd
from FairnessLoss import FairnessLoss

class MCboost():
    def __init__(self):
        p=Preprocess()
        self.f=p.f #predict
        self.origin=p.f.copy()
        self.t=p.t #true
        #self.loss=p.loss #initial loss
        self.levels=p.levels
        self.n=len(self.f)
        self.indexes=np.arange(self.n)
        self.m=10 #each fetch
        self.alpha=(np.max(self.f)-np.min(self.f))/self.levels
        self.lambda_discretization=[(i+1/2)*self.alpha+np.min(self.f) for i in range(self.levels-1)]
        self.I_min=self.n*math.log(self.n)/self.m
        self.individual_loss=[]
        self.group_loss=[]
        self.accuracy=[]
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

    def boost(self):
        print("Running MCboost...........................")
        no_update=0
        while no_update<self.I_min:
            no_update+=1
            C=self.subset(np.random.choice(self.indexes, self.m, replace=False))
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
            df = pd.DataFrame()
            df['calibration_prediction']=self.f
            df['origin']=self.origin
            df['calibration_true']=self.t
            df.to_csv("tmp_comparison.csv", index=False, header=['calibration_prediction', 'origin','calibration_true'])
            f_tmp=FairnessLoss()
            self.individual_loss.append(f_tmp.individual_loss(f_tmp.cali_prediction))
            self.group_loss.append(f_tmp.group_loss(f_tmp.cali_prediction))
            self.accuracy.append(f_tmp.accuracy())

    def run(self):
        self.boost()
              
                
                    
