import pandas as pd
import numpy as np
import collections
import json
import os

from FairnessLoss import FairnessLoss

class Response:
    def __init__(self,args):
        with open(os.path.dirname(os.path.abspath(__file__))+"/../Settings_CUD.json","r") as json_file:
            info = json.load(json_file)
        self.groups=info["groups"]

        self.p_s=args["p_s"]
        self.v=args["v"]
        self.omega=args["omega"]

        self.tpr=args["tpr"]
        self.fpr=args["fpr"]

        self.train_data=pd.read_csv(info["path_to_train_data"],header=0)
        self.level_length=info["level_length"]
        self.candidates=args["candidates"]
        self.feedback=collections.defaultdict(float)
        self.feedback_calculator()

        #return self.feedback

    def equalized_odds_loss(self,fpr:float,tpr:float):
        return fpr-tpr

    def find_groups(self):
        current_group=set()
        for candidate in self.candidates:
            current_group.add(self.train_data["Admission_Channel"][candidate])
        current_group=list(current_group)
        group_length=len(current_group)
        current_best_group=current_group[0]
        current_worst_group=current_group[0]
        current_best=self.equalized_odds_loss(self.fpr[current_best_group],self.tpr[current_best_group])
        current_worst=self.equalized_odds_loss(self.fpr[current_worst_group],self.tpr[current_worst_group])
        if group_length==1:
            return current_group[0],current_group[0]
        else:
            for group in current_group:
                if self.equalized_odds_loss(self.fpr[group],self.tpr[group])>current_best:
                    current_best=self.equalized_odds_loss(self.fpr[group],self.tpr[group])
                    current_best_group=group
                if self.equalized_odds_loss(self.fpr[group],self.tpr[group])<current_worst:
                    current_worst=self.equalized_odds_loss(self.fpr[group],self.tpr[group])
                    current_worst_group=group
            return current_best_group,current_worst_group

    def feedback_calculator(self):
        best_group,worst_group=self.find_groups()
        for candidate in self.candidates:
            if self.train_data["Admission_Channel"][candidate]==best_group:
                self.feedback[candidate]=self.p_s-self.omega if abs(self.p_s+self.omega-self.v)>abs(self.p_s-self.omega-self.v) else self.p_s-self.omega
            elif self.train_data["Admission_Channel"][candidate]==worst_group:
                self.feedback[candidate]=self.p_s+self.omega if abs(self.p_s+self.omega-self.v)>abs(self.p_s-self.omega-self.v) else self.p_s+self.omega
            #else:
            #    self.feedback[candidate]=self.p_s
            
        