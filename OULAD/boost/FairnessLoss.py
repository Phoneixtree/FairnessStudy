import pandas as pd
import numpy as np
import collections
import itertools
import math
import sys
import json

matrix_path="../matrix"
sys.path.append(matrix_path)
from ConfusionMatrix import ConfusionMatrix

class FairnessLoss:
    def __init__(self):
        with open('./BoostSetting_OULAD.json', 'r') as json_file:
            info = json.load(json_file)
        #df=pd.read_csv(self.info["comparison_result"])
        self.level_length=info["level_length"]
        df=pd.read_csv(info["comparison_trail"])
        self.prediction=df["origin_prediction"]
        self.cali_prediction=df["calibrated_prediction"]
        self.actual=df["actual"]
        self.train_data=pd.read_csv(info["train_data"],header=0)
        self.groups=self.division(self.train_data[info["sensitive_attribute"]])
        self.calibration_true=df["actual"]
        self.M=info["data_size"]
        self.fair_factor=info["fair_factor"]
        #self.indi_prediction=df["individual_f"]

    def division(self,fullset): #for one specialized attribute
        divisions=collections.defaultdict(list) #divided by value of this attribute
        for index,value in enumerate(fullset):
            divisions[value].append(index)
        return divisions
       
    def individual_loss(self,target):
        S_comb=list(itertools.combinations(self.groups.keys(), 2))
        F1=np.array([])
        for s in S_comb:
            n1=len(self.groups[s[0]])
            n2=len(self.groups[s[1]])
            tmp=0
            for x1 in range(n1):
                for x2 in range(n2):
                    tmp+=np.abs(self.calibration_true[x1]-self.calibration_true[x2])*((self.fair_factor*(target[x1]-target[x2]))**2)
            F1=np.append(F1,tmp/(n1*n2))
        return np.mean(F1)
    
    def group_loss(self,target):
        S_comb=list(itertools.combinations(self.groups.keys(), 2))
        F2=np.array([])
        for s in S_comb:
            n1=len(self.groups[s[0]])
            n2=len(self.groups[s[1]])
            tmp=0
            for x1 in range(n1):
                for x2 in range(n2):
                    tmp+=np.abs(self.calibration_true[x1]-self.calibration_true[x2])*(self.fair_factor*(target[x1]-target[x2]))
            F2=np.append(F2,(tmp/(n1*n2))**2)
        return np.mean(F2)

    def accuracy(self):
        M=self.M
        length=self.level_length
        def get_grade(gpa):
            if gpa >= 0.8:
                return "Distinction"
            elif gpa >=0.6:
                return "Pass"
            else:
                return "Fail"
        cali_prediction=[get_grade(self.cali_prediction[i]) for i in range(M)]
        calibration_true=[get_grade(self.calibration_true[i]) for i in range(M) ]
        #accu=len([i for i in range(M) if get_grade(self.cali_prediction[i])==get_grade(self.calibration_true[i])])/M
        CM=ConfusionMatrix({"actual":calibration_true,"calibrated_prediction":cali_prediction})
        matrix=CM.matrix
        accu=sum([matrix[i][i] for i in range(length)])/M
        return accu
        
""" f=FairnessLoss()

print(f"before calibration, the individual loss is {f.individual_loss(f.prediction):.4f}")
print(f"before calibration, the group loss is {f.group_loss(f.prediction):.4f}") """
""" print(f"after calibration, the individual loss is {f.individual_loss(f.calibration_true):.4f}")
print(f"after calibration, the group loss is {f.group_loss(f.calibration_true):.4f}") """


