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
        self.p_s=info["p_s"]
        self.v=info["v"]

        self.level_length=info["level_length"]
        self.candidates=args["candidates"]
        self.feedback=collections.defaultdict(float)


    def feedback_calculator(self,):
        best_group=
        worst_group=
        for group in self.groups:
            self.feedback[group]=
        