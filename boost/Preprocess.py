import json
import pandas as pd
import numpy as np
import csv
from collections import Counter

class Preprocess():
    def __init__(self):
        with open('./BoostSetting_CUD.json', 'r') as json_file:
            info = json.load(json_file)
        self.predicted = pd.read_csv(info["prediction"], header=0).values
        self.actual = pd.read_csv(info["actual"], header=0).values
        self.f=[]
        self.t=[]
        self.weights=np.array(info["level_weight"])
        self.levels=info["level_length"] # Number of categories classified
        self.predict()
        self.write_prediction(info["prediction_numerical"])

    def char_to_number(self,char):
        char_map = {
            'A': 3.85, 'B+': 3.5, 'B': 3.15, 'B-': 2.85, 'C': 2.2, 'D': 1.35, 'F': 0
        }
        return char_map.get(char, 0)

    def grade_counter(self,arr):
        counts=Counter(arr)
        return [counts.get(key, 0) for key in ['A', 'B+', 'B', 'B-', 'C', 'D', 'F']]

    def predict(self):
        for line in self.predicted:
            self.f.append(np.dot(np.array(self.grade_counter(line)),self.weights)/self.levels)
        for line in self.actual:
            self.t.append(np.dot(np.array(self.grade_counter(line)),self.weights))

    def write_prediction(self,target):
        df=pd.DataFrame()
        df["predicted"]=self.f
        df["actual"]=self.t
        df.to_csv(target,index=False,header=["predicted","actual"])
        

"""     def loss_func(self):
        weight=np.array([3.85,3.5,3.15,2.85,2.2,1.35,0])
        for index,f in self.f:
            self.loss.append(math.abs(np.dot(weight,np.array(f))-np.dot(weight,np.array(self.t[index])))) """