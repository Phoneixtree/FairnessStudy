from Preprocess import Preprocess
from Boost import Boost
import pandas as pd
import collections

class Individual():
    def __init__(self,info):
        self.attributes=info['attri']
        self.f=info['f']
        self.t=info['t']
        self.levels=info['levels']
        self.n=info['n']
        self.train_data=pd.read_csv('train_data.csv', header=0)
        self.run()
        
    def division(self,fullset): #for one specialized attribute
        divisions=collections.defaultdict(list) #divided by value of this attribute
        for index,value in enumerate(fullset):
            divisions[value].append(index)
        return divisions

    def calibration_update(self,index,result):
        for i,r in enumerate(result):
            self.f[index[i]]=r

    def run(self):
        for a in self.attributes: # for each attribute
            divisions=self.division(self.train_data[a])
            for d in divisions.keys():
                info={'indexes':divisions[d],'f':[self.f[i] for i in divisions[d]],'t':[self.t[i] for i in divisions[d]],'levels':self.levels,'n':self.n}
                self.calibration_update(divisions[d],Boost(info).f)


            


    