import numpy as np
import pandas as pd
import json
import os
class ConfusionMatrix:
    def __init__(self,args):
        with open(os.path.dirname(os.path.abspath(__file__))+"/../Settings_OULAD.json","r") as json_file:
            info=json.load(json_file)
        self.levels=info["levels"]
        self.level_length=info["level_length"]
        self.predicted=args["predicted"]
        self.actual=args["actual"]
        self.comb=np.column_stack((self.actual,self.predicted))

        self.initial_matrix=np.zeros((self.level_length,self.level_length),dtype=int)
        self.relation_matrix = np.array(info["relation_matrix"])
        self.confusion_matrix(self.comb)
        #self.matrix=self.initial_matrix
        self.matrix=np.dot(self.initial_matrix,self.relation_matrix)

    def confusion_matrix(self,combination):
        for c in combination:
            tmpTrue=self.levels.index(c[0])
            tmpPred=self.levels.index(c[1])
            self.initial_matrix[tmpTrue][tmpPred]+=1

    def true_positive_rate(self,matrix):
        tmp_sum=0
        for i in range(self.level_length):
            tp=matrix[i][i]
            fn=np.sum(matrix[i])-tp
            if tp+fn>0:
                tmp_sum+=tp/(tp+fn)

        return tmp_sum/self.level_length
    
    def false_positive_rate(self,matrix):
        tmp_sum=0
        total_sum=np.sum(matrix)
        for i in range(self.level_length):
            tp=matrix[i][i]
            fp=np.sum(matrix[:,i])-tp
            tn=total_sum+tp-np.sum(matrix[i])-np.sum(matrix[:,i])
            if fp+tn>0:
                tmp_sum+=fp/(fp+tn)
        return tmp_sum/self.level_length