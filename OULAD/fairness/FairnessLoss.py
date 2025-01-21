import pandas as pd
import numpy as np
import collections
import itertools
import collections
import os
import json

from ConfusionMatrix import ConfusionMatrix

class FairnessLoss:
    def __init__(self,args):
        with open(os.path.dirname(os.path.abspath(__file__))+"/../Settings_OULAD.json","r") as json_file:
            info = json.load(json_file)
        #df=pd.read_csv(self.info["comparison_result"])
        self.level_length=info["level_length"]
        #df=pd.read_csv(info["path_to_comparison_trail"])
        #self.predicted_origin=df["origin_prediction"]
        #self.predicted=df["calibrated_prediction"]
        #self.actual=df["actual"]
        self.predicted=args["predicted"]
        self.actual=args["actual"]
        self.train_data=pd.read_csv(info["path_to_train_data"],header=0)
        self.groups=self.division(self.train_data[info["sensitive_attribute"]])
        self.tp_rate=collections.defaultdict(float)
        self.fp_rate=collections.defaultdict(float)
        self.zoom_factor=info["zoom_factor"]
        #self.calibration_true=df["actual"]
        self.m=info["data_size"]
        self.run()

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
    
    def true_positive_rate(self,matrix):
        tmp_sum=0
        for i in range(self.level_length):
            tp=matrix[i][i]
            fn=np.sum(matrix[i])-tp
            if tp+fn>0:
                tmp_sum+=tp/(tp+fn)
            else:
                tmp_sum+=0
        return tmp_sum/self.level_length
    
    def false_positive_rate(self,matrix):
        tmp_sum=0
        for i in range(self.level_length):
            tp=matrix[i][i]
            fp=np.sum(matrix[:,i])-tp
            tn=np.sum(matrix)+tp-np.sum(matrix[i])-np.sum(matrix[:,i])
            if fp+tn>0:
                tmp_sum+=fp/(fp+tn)
            else:
                tmp_sum+=0
        return tmp_sum/self.level_length

    def get_grade(self,gpa):
        gpa/=self.zoom_factor
        if gpa>= 0.862:
            return "Distinction"
        elif gpa>=0.108:
            return "Pass"
        else:
            return "Fail"

    """    def accuracy(self):
        m=self.m
        length=self.level_length
        cali_prediction=[get_grade(self.cali_prediction[i]) for i in range(M)]
        calibration_true=[get_grade(self.calibration_true[i]) for i in range(M) ]
        #accu=len([i for i in range(M) if get_grade(self.cali_prediction[i])==get_grade(self.calibration_true[i])])/M
        CM=ConfusionMatrix({"actual":calibration_true,"calibrated_prediction":cali_prediction})
        matrix=CM.matrix
        accu=sum([matrix[i][i] for i in range(length)])/m
        return accu """
        
    def run(self):
        for i, (key, indexes) in enumerate(self.groups.items()):
            tmp_length = len(indexes)
            tmp_actual = [self.get_grade(self.actual[i]) for i in indexes]
            tmp_predicted = [self.get_grade(self.predicted[i]) for i in indexes]
            tmp_confusionmatrix = ConfusionMatrix({"actual": tmp_actual, "predicted": tmp_predicted})
            #total = [sum(tmp_confusionmatrix.matrix[i]) for i in range(self.level_length)]
            self.tp_rate[key] = self.true_positive_rate(tmp_confusionmatrix.matrix)
            self.fp_rate[key] = self.false_positive_rate(tmp_confusionmatrix.matrix)
        #print(self.tp_rate)
        """ total_actual = [self.get_grade(self.actual[i]) for i in range(self.m)]
        total_predicted = [self.get_grade(self.predicted[i]) for i in range(self.m)]
        total_confusionmatrix = ConfusionMatrix({"actual": total_actual, "predicted": total_predicted})
        class_accuracies = []
        for i in range(self.level_length):
            tp_tn = total_confusionmatrix.matrix[i][i]
            fn_fp = sum(total_confusionmatrix.matrix[i]) + sum(total_confusionmatrix.matrix[:, i]) - tp_tn
            class_accuracy = (tp_tn + (self.m - fn_fp)) / self.m
            class_accuracies.append(class_accuracy)

        self.accuracy = sum(class_accuracies) / self.level_length """



""" f=FairnessLoss()

print(f"before calibration, the individual loss is {f.individual_loss(f.prediction):.4f}")
print(f"before calibration, the group loss is {f.group_loss(f.prediction):.4f}") """
""" print(f"after calibration, the individual loss is {f.individual_loss(f.calibration_true):.4f}")
print(f"after calibration, the group loss is {f.group_loss(f.calibration_true):.4f}") """


