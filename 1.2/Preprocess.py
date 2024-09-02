import pandas as pd
import numpy as np
import math
import csv
from collections import Counter

class Preprocess():
    def __init__(self):
        self.predictions = pd.read_csv('predict_outcome.csv', header=0).values
        self.truelevel = pd.read_csv("True_level.csv", header=0).values
        """ self.predictions = pd.read_csv('p_test.csv', header=0).values
        self.truelevel = pd.read_csv("t_test.csv", header=0).values """
        self.f=[]
        self.t=[]
        self.weight=np.array([3.85,3.5,3.15,2.85,2.2,1.35,0])
        self.levels=len(self.weight)*2 # Number of categories classified
        #self.loss=[]
        self.predic()
        #self.loss_func()

    def char_to_number(self,char):
        char_map = {
            'A': 3.85, 'B+': 3.5, 'B': 3.15, 'B-': 2.85, 'C': 2.2, 'D': 1.35, 'F': 0
        }
        return char_map.get(char, 0)

    def grade_counter(self,arr):
        counts=Counter(arr)
        return [counts.get(key, 0) for key in ['A', 'B+', 'B', 'B-', 'C', 'D', 'F']]

    def predic(self):
        for line in self.predictions:
            self.f.append(np.dot(np.array(self.grade_counter(line)),self.weight)/len(self.weight))
        for line in self.truelevel:
            self.t.append(np.dot(np.array(self.grade_counter(line)),self.weight))
        

"""     def loss_func(self):
        weight=np.array([3.85,3.5,3.15,2.85,2.2,1.35,0])
        for index,f in self.f:
            self.loss.append(math.abs(np.dot(weight,np.array(f))-np.dot(weight,np.array(self.t[index])))) """