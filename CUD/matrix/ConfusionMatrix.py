import numpy as np
import pandas as pd
import json
import os
class ConfusionMatrix:
    def __init__(self,args):
        with open(os.path.dirname(os.path.abspath(__file__))+"/../Settings_CUD.json","r") as json_file:
            info=json.load(json_file)
        self.levels=info["levels"]
        self.level_length=info["level_length"]
        self.predicted=args["predicted"]
        self.actual=args["actual"]
        self.comb=np.column_stack((self.actual,self.predicted))

        self.initial_matrix=np.zeros((self.level_length,self.level_length),dtype=int)
        #self.relation_matrix = np.array(info["relation_matrix"])
        self.confusion_matrix(self.comb)
        self.matrix=self.initial_matrix
        #self.matrix=np.dot(self.initial_matrix,self.relation_matrix)

    def confusion_matrix(self,combination):
        for c in combination:
            tmpTrue=self.levels.index(c[0])
            tmpPred=self.levels.index(c[1])
            self.initial_matrix[tmpTrue][tmpPred]+=1