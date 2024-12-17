import json
import pandas as pd
import numpy as np
import os
from collections import Counter

class Preprocess():
    def __init__(self):
        with open(os.path.dirname(os.path.abspath(__file__))+"/../Settings_CUD.json","r") as json_file:
            info = json.load(json_file)
        self.predicted = pd.read_csv(info["path_to_predicted"], header=0).values
        self.actual = pd.read_csv(info["path_to_actual"], header=0).values
        self.f=[]
        self.t=[]
        self.zoom_factor=info["zoom_factor"]
        self.level_weight=np.array(info["level_weight"])
        self.levels=info["levels"]
        self.level_length=info["level_length"] # Number of categories classified
        self.make_prediction()
        self.write_prediction(info["path_to_predicted_numerical"])

    def char_to_number(self,char):
        char_map = dict(zip(self.levels,self.level_weight))
        return char_map.get(char, 0)

    def grade_counter(self,arr):
        counts=Counter(arr)
        return [counts.get(key, 0) for key in self.levels]

    def make_prediction(self):
        for line in self.predicted:
            self.f.append(np.dot(np.array(self.grade_counter(line)),self.level_weight)*self.zoom_factor/self.level_length)
        for line in self.actual:
            self.t.append(np.dot(np.array(self.grade_counter(line)),self.level_weight)*self.zoom_factor)

    def write_prediction(self,target):
        df=pd.DataFrame()
        df["predicted"]=self.f
        df["actual"]=self.t
        df.to_csv(target,index=False,header=["predicted","actual"])