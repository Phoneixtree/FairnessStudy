import itertools
import math
import json
import os
import numpy as np
import pandas as pd

from FairnessLoss import FairnessLoss
from Response import Response

class MCboost():
    def __init__(self):
        with open(os.path.dirname(os.path.abspath(__file__))+"/../Settings_OULAD.json","r") as json_file:
            info = json.load(json_file)
        preprocessed=pd.read_csv(info["path_to_predicted_numerical"])
        self.predicted=preprocessed["predicted"] #prediction
        self.predicted_origin=self.predicted.copy()
        self.actual=preprocessed["actual"] #true
        #self.sorted_actual=self.actual[:]
        #self.sorted_actual.sort_values()

        self.level_length=info["level_length"]
        self.n=info["data_size"]
        self.indexes=np.arange(self.n)
        self.m=info["fetch_size"] #each fetch
        self.alpha=info["alpha"]
        self.lambda_discretization=[(i+1/2)*self.alpha for i in range(int(1/self.alpha-1))]
        #self.I_min=self.n*math.log(self.n)/self.m #iteration
        self.I_min=40
        self.progressive_parameters=0.5

        self.comparison_trail=info["path_to_comparison_trail"]
        self.tpr_trail={i:[0] for i in range(13)}
        self.fpr_trail={i:[0] for i in range(13)}
        self.accuracy_trail=[]

        self.run()

    def discretization(self,S):
        S_v={}
        for l in self.lambda_discretization:
            s_lambda=[index for index,f in enumerate(self.predicted) if l-self.alpha/2<=f<l+self.alpha/2 and index in S]
            S_v[l]=s_lambda
        return S_v

    def subset(self,set):
        tmp=[]
        for n in range(1,len(set)+1):
            tmp.extend(itertools.combinations(set,n))
        return tmp

    #def find_pstar(self,value):
    #    return np.searchsorted(self.sorted_actual, value)/self.n

    def guess_and_check(self,S,v,omega):
        #p_s=sum([self.find_pstar(self.actual[s]) for s in S])/len(S)
        p_s=sum([(self.actual[s]) for s in S])/len(S)
        if math.fabs(p_s-v)<2*omega:
            return [True,0]
        else:
            #tpr_tmp={1:self.tpr_trail[1][-1],3:self.tpr_trail[3][-1],4:self.tpr_trail[4][-1],7:self.tpr_trail[7][-1]}
            #fpr_tmp={1:self.fpr_trail[1][-1],3:self.fpr_trail[3][-1],4:self.fpr_trail[4][-1],7:self.fpr_trail[7][-1]}
            tpr_tmp={i:self.tpr_trail[i][-1] for i in range(13)}
            fpr_tmp={i:self.fpr_trail[i][-1] for i in range(13)}
            response=Response({"p_s":p_s,"v":v,"omega":omega,"tpr":tpr_tmp,"fpr":fpr_tmp,"candidates":S})
            return [False,p_s-omega,response.feedback]

    def boost(self):
        print("Running MCboost...........................")
        #iteration=0
        no_update=0
        #while no_update<self.I_min and self.overflow_check():
        while no_update<self.I_min:
            no_update+=1
            #iteration+=1
            print(f"Currently Nonupdate MCboost iteration {no_update}.")
            C=self.subset(np.random.choice(self.indexes, self.m, replace=False))
            #tmp_f=self.predicted.copy()
            for S in C:
                S_v=self.discretization(S)
                for key in self.lambda_discretization:
                    S_v_current=S_v[key]
                    """ if len(S_v_current)<(self.alpha**2)*len(S):
                        continue """
                    if len(S_v_current)==0:
                        #print("empty")
                        for s in S_v_current:
                            self.predicted[s]+=r[2][s]
                        continue
                    v_average=sum([self.predicted[s] for s in S_v_current])/len(S_v_current)
                    r=self.guess_and_check(S_v_current,v_average,self.alpha/4)
                    #print(r)
                    if not r[0]:
                        no_update=0
                        for s in S_v_current:
                            self.predicted[s]+=self.progressive_parameters*(r[1]-v_average)
                            #self.predicted[s]+=r[1]-v_average
                        
            fair_args={"predicted":self.predicted, "actual":self.actual}
            fairnessloss=FairnessLoss(fair_args)
            for key in self.tpr_trail:
                self.tpr_trail[key].append(fairnessloss.tp_rate[key])
                self.fpr_trail[key].append(fairnessloss.fp_rate[key])
            #self.accuracy_trail.append(fairnessloss.accuracy)

    def run(self):
        self.boost()