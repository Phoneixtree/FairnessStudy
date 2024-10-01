import numpy as np
import pandas as pd
from collections import Counter

df1=pd.read_csv("./train_data2.csv")

df2=df1["final_result"]

df2.to_csv("./True_level2.csv", index=False, header=['Level'])

