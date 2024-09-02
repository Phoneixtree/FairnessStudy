import numpy as np
import pandas as pd
from collections import Counter

source=pd.read_csv('train_data1.csv', header=0).values

admission=[s[1] for s in source]

counter=Counter(admission)

print(counter)
