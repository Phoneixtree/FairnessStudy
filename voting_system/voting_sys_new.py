# -*- coding: utf-8 -*-
"""

"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import csv
import random
import math
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


#导入所需要的包
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
import xgboost as xgb


from matplotlib.colors import ListedColormap
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV #网格搜索
import seaborn as sns#绘图包

from sklearn import preprocessing

# 忽略警告
import warnings
warnings.filterwarnings("ignore")

"""导入训练数据集（是CUHK还是OULAD）"""

# dataset = pd.read_csv('train_data1.csv')
dataset = pd.read_csv('train_data1.csv')
dataset.head()

"""选择X和y分别代表的对象（对于CUHK数据集选择1-6，OULAD数据集是1-9）"""

X = dataset.iloc[:,1:6].values
y = dataset['Level'].values
#print(type((x)))


add_col_y = []
for i in range(len(y)):
    if y[i] == 'A':
        tmp = [1]
        add_col_y.append(tmp)
    if y[i] == 'B+':
        tmp = [2]
        add_col_y.append(tmp)
    if y[i] == 'B':
        tmp = [3]
        add_col_y.append(tmp)
    if y[i] == 'B-':
        tmp = [4]
        add_col_y.append(tmp)
    if y[i] == 'C':
        tmp = [5]
        add_col_y.append(tmp)
    if y[i] == 'D':
        tmp = [6]
        add_col_y.append(tmp)
    if y[i] == 'F':
        tmp = [7]
        add_col_y.append(tmp)

y = np.array(add_col_y)
#print(y)

mm = preprocessing.MinMaxScaler()

#  所有数据集归一化处理

X = mm.fit_transform(X)

#####



##### 随机重组数据集 X y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.001, random_state = 0)


#min_max_scaler = preprocessing.normalize(norm='l2')#默认为范围0~1，拷贝操作

#X_train = min_max_scaler.fit_transform(X_train)
#X_test = min_max_scaler.fit_transform(X_test)



#####

def func_divide_datasetX(l,r):
    new_train_set = []
    new_train_set = X_train[l:r]
    new_train_set = np.array(new_train_set)
    return new_train_set

def func_divide_datasetY(l,r):
    new_train_set = []
    new_train_set = y_train[l:r]
    new_train_set = np.array(new_train_set)
    return new_train_set


"### nn多层感知神经网络"
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(solver='sgd', alpha=1e-5,hidden_layer_sizes=(50, 50), random_state=1)

"### 第一组重划分 1~109 数据为测试集"
test_x1 = func_divide_datasetX(0,109)
test_y1 = func_divide_datasetY(0,109)
train_x1 = func_divide_datasetX(109,544)
train_y1 = func_divide_datasetY(109,544)
model = mlp.fit(train_x1, train_y1)
y_pred1 = model.predict(test_x1)
add_col1 = [] # 第一组的新增特征值
for i in range(len(y_pred1)):
    tmp = y_pred1[i]
    add_col1.append(tmp)

"### 第二组重划分 110~218 数据为测试集"

test_x1 = func_divide_datasetX(109,218)
test_y1 = func_divide_datasetY(109,218)
train_x1 = np.concatenate((func_divide_datasetX(0,109),func_divide_datasetX(218,544)),axis=0)
train_y1 = np.concatenate((func_divide_datasetY(0,109),func_divide_datasetY(218,544)),axis=0)
model = mlp.fit(train_x1, train_y1)
y_pred2 = model.predict(test_x1)
add_col2 = [] # 第二组的新增特征值
for i in range(len(y_pred2)):
    tmp = y_pred2[i]
    add_col2.append(tmp)

"### 第三组重划分 219~327 数据为测试集"

test_x1 = func_divide_datasetX(218,327)
test_y1 = func_divide_datasetY(218,327)
train_x1 = np.concatenate((func_divide_datasetX(0,218),func_divide_datasetX(327,544)),axis=0)
train_y1 = np.concatenate((func_divide_datasetY(0,218),func_divide_datasetY(327,544)),axis=0)
model = mlp.fit(train_x1, train_y1)
y_pred3 = model.predict(test_x1)
add_col3 = [] # 第三组的新增特征值
for i in range(len(y_pred3)):
    tmp = y_pred3[i]
    add_col3.append(tmp)

"### 第四组重划分 328~436 数据为测试集"

test_x1 = func_divide_datasetX(327,436)
test_y1 = func_divide_datasetY(327,436)
train_x1 = np.concatenate((func_divide_datasetX(0,327),func_divide_datasetX(436,544)),axis=0)
train_y1 = np.concatenate((func_divide_datasetY(0,327),func_divide_datasetY(436,544)),axis=0)
model = mlp.fit(train_x1, train_y1)
y_pred4 = model.predict(test_x1)
add_col4 = [] # 第四组的新增特征值
for i in range(len(y_pred4)):
    tmp = y_pred4[i]
    add_col4.append(tmp)

"### 第五组重划分 437~544 数据为测试集"

test_x1 = func_divide_datasetX(436,544)
test_y1 = func_divide_datasetY(436,544)
train_x1 = func_divide_datasetX(0,436)
train_y1 = func_divide_datasetY(0,436)
model = mlp.fit(train_x1, train_y1)
y_pred5 = model.predict(test_x1)
add_col5 = [] # 第四组的新增特征值
for i in range(len(y_pred5)):
    tmp = y_pred5[i]
    add_col5.append(tmp)


"# 合并knn模型产生的预测数据 作为新的特征值"

add_col_nn = add_col1 + add_col2 + add_col3 + add_col4 + add_col5
add_col_NN = []
print(len(add_col_nn))
for i in range(len(add_col_nn)):
    if add_col_nn[i] == 1:
        tmp = [1]
        add_col_NN.append(tmp)
    if add_col_nn[i] == 2:
        tmp = [2]
        add_col_NN.append(tmp)
    if add_col_nn[i] == 3:
        tmp = [3]
        add_col_NN.append(tmp)
    if add_col_nn[i] == 4:
        tmp = [4]
        add_col_NN.append(tmp)
    if add_col_nn[i] == 5:
        tmp = [5]
        add_col_NN.append(tmp)
    if add_col_nn[i] == 6:
        tmp = [6]
        add_col_NN.append(tmp)
    if add_col_nn[i] == 7:
        tmp = [7]
        add_col_NN.append(tmp)




#add_col_NN = preprocessing.normalize(add_col_NN, norm='l2')
add_col_NN = mm.fit_transform(add_col_NN)                                      
add_col_NN = np.array(add_col_NN)

#print(add_col_NN)

"# knn 重新划分数据集 赋予新的特征值"

import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
import  matplotlib.pyplot as plt

knn=KNeighborsClassifier(algorithm='auto', leaf_size=5,
                                            metric='minkowski',
                                            metric_params=None, n_jobs=None,
                                            n_neighbors=8, p=2,
                                            weights='uniform') #引入训练方法
"### 第一组重划分 1~109 数据为测试集"
test_x1 = func_divide_datasetX(0,109)
test_y1 = func_divide_datasetY(0,109)
train_x1 = func_divide_datasetX(109,544)
train_y1 = func_divide_datasetY(109,544)
model = knn.fit(train_x1, train_y1)
y_pred1 = model.predict(test_x1)
add_col1 = [] # 第一组的新增特征值
for i in range(len(y_pred1)):
    tmp = y_pred1[i]
    add_col1.append(tmp)

"### 第二组重划分 110~218 数据为测试集"

test_x1 = func_divide_datasetX(109,218)
test_y1 = func_divide_datasetY(109,218)
train_x1 = np.concatenate((func_divide_datasetX(0,109),func_divide_datasetX(218,544)),axis=0)
train_y1 = np.concatenate((func_divide_datasetY(0,109),func_divide_datasetY(218,544)),axis=0)
model = knn.fit(train_x1, train_y1)
y_pred2 = model.predict(test_x1)
add_col2 = [] # 第二组的新增特征值
for i in range(len(y_pred2)):
    tmp = y_pred2[i]
    add_col2.append(tmp)

"### 第三组重划分 219~327 数据为测试集"

test_x1 = func_divide_datasetX(218,327)
test_y1 = func_divide_datasetY(218,327)
train_x1 = np.concatenate((func_divide_datasetX(0,218),func_divide_datasetX(327,544)),axis=0)
train_y1 = np.concatenate((func_divide_datasetY(0,218),func_divide_datasetY(327,544)),axis=0)
model = knn.fit(train_x1, train_y1)
y_pred3 = model.predict(test_x1)
add_col3 = [] # 第三组的新增特征值
for i in range(len(y_pred3)):
    tmp = y_pred3[i]
    add_col3.append(tmp)

"### 第四组重划分 328~436 数据为测试集"

test_x1 = func_divide_datasetX(327,436)
test_y1 = func_divide_datasetY(327,436)
train_x1 = np.concatenate((func_divide_datasetX(0,327),func_divide_datasetX(436,544)),axis=0)
train_y1 = np.concatenate((func_divide_datasetY(0,327),func_divide_datasetY(436,544)),axis=0)
model = knn.fit(train_x1, train_y1)
y_pred4 = model.predict(test_x1)
add_col4 = [] # 第四组的新增特征值
for i in range(len(y_pred4)):
    tmp = y_pred4[i]
    add_col4.append(tmp)

"### 第五组重划分 437~544 数据为测试集"

test_x1 = func_divide_datasetX(436,544)
test_y1 = func_divide_datasetY(436,544)
train_x1 = func_divide_datasetX(0,436)
train_y1 = func_divide_datasetY(0,436)
model = knn.fit(train_x1, train_y1)
y_pred5 = model.predict(test_x1)
add_col5 = [] # 第四组的新增特征值
for i in range(len(y_pred5)):
    tmp = y_pred5[i]
    add_col5.append(tmp)


"# 合并knn模型产生的预测数据 作为新的特征值"

add_col_knn = add_col1 + add_col2 + add_col3 + add_col4 + add_col5
add_col_KNN = []
for i in range(len(add_col_knn)):
    if add_col_knn[i] == 1:
        tmp = [1]
        add_col_KNN.append(tmp)
    if add_col_knn[i] == 2:
        tmp = [2]
        add_col_KNN.append(tmp)
    if add_col_knn[i] == 3:
        tmp = [3]
        add_col_KNN.append(tmp)
    if add_col_knn[i] == 4:
        tmp = [4]
        add_col_KNN.append(tmp)
    if add_col_knn[i] == 5:
        tmp = [5]
        add_col_KNN.append(tmp)
    if add_col_knn[i] == 6:
        tmp = [6]
        add_col_KNN.append(tmp)
    if add_col_knn[i] == 7:
        tmp = [7]
        add_col_KNN.append(tmp)




add_col_KNN = preprocessing.normalize(add_col_KNN, norm='l2')
#add_col_KNN = preprocessing.StandardScaler().fit_transform(add_col_KNN)                                      
add_col_KNN = np.array(add_col_KNN)


#X_train = np.append(X_train, add_col_KNN, axis=1)


cnt = 0
for i in range(len(test_y1)):
    if test_y1[i] == y_pred5[i]:
        cnt = cnt + 1




#####################################################################

"## GB重新划分数据集 赋予新的特征值"

from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(min_samples_split = 5,n_estimators=100, min_samples_leaf = 32,learning_rate=0.1, 
                                 max_depth = 7,max_features='sqrt', subsample = 1.0,random_state = 0 )



"### 第一组重划分 1~109 数据为测试集"

test_x1 = func_divide_datasetX(0,109)
test_y1 = func_divide_datasetY(0,109)
train_x1 = func_divide_datasetX(109,544)
train_y1 = func_divide_datasetY(109,544)
model = clf.fit(train_x1, train_y1)
y_pred1 = model.predict(test_x1)
add_col1 = [] # 第一组的新增特征值
for i in range(len(y_pred1)):
    tmp = y_pred1[i]
    add_col1.append(tmp)

"### 第二组重划分 110~218 数据为测试集"

test_x1 = func_divide_datasetX(109,218)
test_y1 = func_divide_datasetY(109,218)
train_x1 = np.concatenate((func_divide_datasetX(0,109),func_divide_datasetX(218,544)),axis=0)
train_y1 = np.concatenate((func_divide_datasetY(0,109),func_divide_datasetY(218,544)),axis=0)
model = clf.fit(train_x1, train_y1)
y_pred2 = model.predict(test_x1)
add_col2 = [] # 第二组的新增特征值
for i in range(len(y_pred2)):
    tmp = y_pred2[i]
    add_col2.append(tmp)

"### 第三组重划分 219~327 数据为测试集"

test_x1 = func_divide_datasetX(218,327)
test_y1 = func_divide_datasetY(218,327)
train_x1 = np.concatenate((func_divide_datasetX(0,218),func_divide_datasetX(327,544)),axis=0)
train_y1 = np.concatenate((func_divide_datasetY(0,218),func_divide_datasetY(327,544)),axis=0)
model = clf.fit(train_x1, train_y1)
y_pred3 = model.predict(test_x1)
add_col3 = [] # 第三组的新增特征值
for i in range(len(y_pred3)):
    tmp = y_pred3[i]
    add_col3.append(tmp)

"### 第四组重划分 328~436 数据为测试集"

test_x1 = func_divide_datasetX(327,436)
test_y1 = func_divide_datasetY(327,436)
train_x1 = np.concatenate((func_divide_datasetX(0,327),func_divide_datasetX(436,544)),axis=0)
train_y1 = np.concatenate((func_divide_datasetY(0,327),func_divide_datasetY(436,544)),axis=0)
model = clf.fit(train_x1, train_y1)
y_pred4 = model.predict(test_x1)
add_col4 = [] # 第四组的新增特征值
for i in range(len(y_pred4)):
    tmp = y_pred4[i]
    add_col4.append(tmp)

"### 第五组重划分 437~544 数据为测试集"

test_x1 = func_divide_datasetX(436,544)
test_y1 = func_divide_datasetY(436,544)
train_x1 = func_divide_datasetX(0,436)
train_y1 = func_divide_datasetY(0,436)
model = clf.fit(train_x1, train_y1)
y_pred5 = model.predict(test_x1)
add_col5 = [] # 第四组的新增特征值
for i in range(len(y_pred5)):
    tmp = y_pred5[i]
    add_col5.append(tmp)


"# 合并GB模型产生的预测数据 作为新的特征值"

add_col_gb = add_col1 + add_col2 + add_col3 + add_col4 + add_col5
add_col_GB = []
for i in range(len(add_col_gb)):
    if add_col_gb[i] == 1:
        tmp = [1]
        add_col_GB.append(tmp)
    if add_col_gb[i] == 2:
        tmp = [2]
        add_col_GB.append(tmp)
    if add_col_gb[i] == 3:
        tmp = [3]
        add_col_GB.append(tmp)
    if add_col_gb[i] == 4:
        tmp = [4]
        add_col_GB.append(tmp)
    if add_col_gb[i] == 5:
        tmp = [5]
        add_col_GB.append(tmp)
    if add_col_gb[i] == 6:
        tmp = [6]
        add_col_GB.append(tmp)
    if add_col_gb[i] == 7:
        tmp = [7]
        add_col_GB.append(tmp)

#add_col_GB = preprocessing.StandardScaler().fit_transform(add_col_GB)
#add_col_GB = preprocessing.RobustScaler().fit_transform(add_col_GB)

add_col_GB = np.array(add_col_GB)


##############################################################################

"####### NB 重新划分数据集 赋予新的特征值"

import pickle
from sklearn.naive_bayes import MultinomialNB
nvclassifier = MultinomialNB()
#MultinomialNB  
"### 第一组重划分 1~109 数据为测试集"

test_x1 = func_divide_datasetX(0,109)
test_y1 = func_divide_datasetY(0,109)
train_x1 = func_divide_datasetX(109,544)
train_y1 = func_divide_datasetY(109,544)
model = nvclassifier.fit(train_x1, train_y1)
y_pred1 = model.predict(test_x1)
add_col1 = [] # 第一组的新增特征值
for i in range(len(y_pred1)):
    tmp = y_pred1[i]
    add_col1.append(tmp)

"### 第二组重划分 110~218 数据为测试集"

test_x1 = func_divide_datasetX(109,218)
test_y1 = func_divide_datasetY(109,218)
train_x1 = np.concatenate((func_divide_datasetX(0,109),func_divide_datasetX(218,544)),axis=0)
train_y1 = np.concatenate((func_divide_datasetY(0,109),func_divide_datasetY(218,544)),axis=0)
model = nvclassifier.fit(train_x1, train_y1)
y_pred2 = model.predict(test_x1)
add_col2 = [] # 第二组的新增特征值
for i in range(len(y_pred2)):
    tmp = y_pred2[i]
    add_col2.append(tmp)

"### 第三组重划分 219~327 数据为测试集"

test_x1 = func_divide_datasetX(218,327)
test_y1 = func_divide_datasetY(218,327)
train_x1 = np.concatenate((func_divide_datasetX(0,218),func_divide_datasetX(327,544)),axis=0)
train_y1 = np.concatenate((func_divide_datasetY(0,218),func_divide_datasetY(327,544)),axis=0)
model = nvclassifier.fit(train_x1, train_y1)
y_pred3 = model.predict(test_x1)
add_col3 = [] # 第三组的新增特征值
for i in range(len(y_pred3)):
    tmp = y_pred3[i]
    add_col3.append(tmp)

"### 第四组重划分 328~436 数据为测试集"

test_x1 = func_divide_datasetX(327,436)
test_y1 = func_divide_datasetY(327,436)
train_x1 = np.concatenate((func_divide_datasetX(0,327),func_divide_datasetX(436,544)),axis=0)
train_y1 = np.concatenate((func_divide_datasetY(0,327),func_divide_datasetY(436,544)),axis=0)
model = nvclassifier.fit(train_x1, train_y1)
y_pred4 = model.predict(test_x1)
add_col4 = [] # 第四组的新增特征值
for i in range(len(y_pred4)):
    tmp = y_pred4[i]
    add_col4.append(tmp)

"### 第五组重划分 437~544 数据为测试集"

test_x1 = func_divide_datasetX(436,544)
test_y1 = func_divide_datasetY(436,544)
train_x1 = func_divide_datasetX(0,436)
train_y1 = func_divide_datasetY(0,436)
model = nvclassifier.fit(train_x1, train_y1)
y_pred5 = model.predict(test_x1)
add_col5 = [] # 第5组的新增特征值
for i in range(len(y_pred5)):
    tmp = y_pred5[i]
    add_col5.append(tmp)


"# 合并NB模型产生的预测数据 作为新的特征值"

add_col_nb = add_col1 + add_col2 + add_col3 + add_col4 + add_col5
add_col_NB = []


for i in range(len(add_col_nb)):
    if add_col_nb[i] == 1:
        tmp = [1]
        add_col_NB.append(tmp)
    if add_col_nb[i] == 2:
        tmp = [2]
        add_col_NB.append(tmp)
    if add_col_nb[i] == 3:
        tmp = [3]
        add_col_NB.append(tmp)
    if add_col_nb[i] == 4:
        tmp = [4]
        add_col_NB.append(tmp)
    if add_col_nb[i] == 5:
        tmp = [5]
        add_col_NB.append(tmp)
    if add_col_nb[i] == 6:
        tmp = [6]
        add_col_NB.append(tmp)
    if add_col_nb[i] == 7:
        tmp = [7]
        add_col_NB.append(tmp)
        

add_col_NB = preprocessing.normalize(add_col_NB, norm='l2')     
#scaler =  preprocessing.StandardScaler() # 然后生成一个标准化对象
#add_col_NB = scaler.fit_transform(add_col_NB)  #然后对数据进行转换
add_col_NB = np.array(add_col_NB)



##############################################################################

"####### SVC 重新划分数据集 赋予新的特征值"

from sklearn.svm import SVC
svcclassifier = SVC(kernel = 'rbf', random_state = 0,C = 0.1)


"### 第一组重划分 1~109 数据为测试集"
test_x1 = func_divide_datasetX(0,109)
test_y1 = func_divide_datasetY(0,109)
train_x1 = func_divide_datasetX(109,544)
train_y1 = func_divide_datasetY(109,544)
model = svcclassifier.fit(train_x1, train_y1)
y_pred1 = model.predict(test_x1)
add_col1 = [] # 第一组的新增特征值
for i in range(len(y_pred1)):
    tmp = y_pred1[i]
    add_col1.append(tmp)

"### 第二组重划分 110~218 数据为测试集"

test_x1 = func_divide_datasetX(109,218)
test_y1 = func_divide_datasetY(109,218)
train_x1 = np.concatenate((func_divide_datasetX(0,109),func_divide_datasetX(218,544)),axis=0)
train_y1 = np.concatenate((func_divide_datasetY(0,109),func_divide_datasetY(218,544)),axis=0)
model = svcclassifier.fit(train_x1, train_y1)
y_pred2 = model.predict(test_x1)
add_col2 = [] # 第二组的新增特征值
for i in range(len(y_pred2)):
    tmp = y_pred2[i]
    add_col2.append(tmp)

"### 第三组重划分 219~327 数据为测试集"

test_x1 = func_divide_datasetX(218,327)
test_y1 = func_divide_datasetY(218,327)
train_x1 = np.concatenate((func_divide_datasetX(0,218),func_divide_datasetX(327,544)),axis=0)
train_y1 = np.concatenate((func_divide_datasetY(0,218),func_divide_datasetY(327,544)),axis=0)
model = svcclassifier.fit(train_x1, train_y1)
y_pred3 = model.predict(test_x1)
add_col3 = [] # 第三组的新增特征值
for i in range(len(y_pred3)):
    tmp = y_pred3[i]
    add_col3.append(tmp)

"### 第四组重划分 328~436 数据为测试集"

test_x1 = func_divide_datasetX(327,436)
test_y1 = func_divide_datasetY(327,436)
train_x1 = np.concatenate((func_divide_datasetX(0,327),func_divide_datasetX(436,544)),axis=0)
train_y1 = np.concatenate((func_divide_datasetY(0,327),func_divide_datasetY(436,544)),axis=0)
model = svcclassifier.fit(train_x1, train_y1)
y_pred4 = model.predict(test_x1)
add_col4 = [] # 第四组的新增特征值
for i in range(len(y_pred4)):
    tmp = y_pred4[i]
    add_col4.append(tmp)

"### 第五组重划分 437~544 数据为测试集"

test_x1 = func_divide_datasetX(436,544)
test_y1 = func_divide_datasetY(436,544)
train_x1 = func_divide_datasetX(0,436)
train_y1 = func_divide_datasetY(0,436)
model = svcclassifier.fit(train_x1, train_y1)
y_pred5 = model.predict(test_x1)
add_col5 = [] # 第四组的新增特征值
for i in range(len(y_pred5)):
    tmp = y_pred5[i]
    add_col5.append(tmp)

"# 合并SVC模型产生的预测数据 作为新的特征值"

add_col_svc = add_col1 + add_col2 + add_col3 + add_col4 + add_col5
add_col_SVC = []
for i in range(len(add_col_svc)):
    if add_col_svc[i] == 1:
        tmp = [1]
        add_col_SVC.append(tmp)
    if add_col_svc[i] == 2:
        tmp = [2]
        add_col_SVC.append(tmp)
    if add_col_svc[i] == 3:
        tmp = [3]
        add_col_SVC.append(tmp)
    if add_col_svc[i] == 4:
        tmp = [4]
        add_col_SVC.append(tmp)
    if add_col_svc[i] == 5:
        tmp = [5]
        add_col_SVC.append(tmp)
    if add_col_svc[i] == 6:
        tmp = [6]
        add_col_SVC.append(tmp)
    if add_col_svc[i] == 7:
        tmp = [7]
        add_col_SVC.append(tmp)

#add_col_SVC = mm.fit_transform(add_col_SVC)  
add_col_SVC = preprocessing.StandardScaler().fit_transform(add_col_SVC)  
add_col_SVC = np.array(add_col_SVC)

##############################################################################


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, max_depth = 7,max_leaf_nodes = 16,
                                    criterion = 'entropy', min_samples_split= 5,min_samples_leaf = 3,random_state = 0)

"####### RF 重新划分数据集 赋予新的特征值"

"### 第一组重划分 1~109 数据为测试集"
test_x1 = func_divide_datasetX(0,109)
test_y1 = func_divide_datasetY(0,109)
train_x1 = func_divide_datasetX(109,544)
train_y1 = func_divide_datasetY(109,544)
model = classifier.fit(train_x1, train_y1)
y_pred1 = model.predict(test_x1)
add_col1 = [] # 第一组的新增特征值
for i in range(len(y_pred1)):
    tmp = y_pred1[i]
    add_col1.append(tmp)

"### 第二组重划分 110~218 数据为测试集"

test_x1 = func_divide_datasetX(109,218)
test_y1 = func_divide_datasetY(109,218)
train_x1 = np.concatenate((func_divide_datasetX(0,109),func_divide_datasetX(218,544)),axis=0)
train_y1 = np.concatenate((func_divide_datasetY(0,109),func_divide_datasetY(218,544)),axis=0)
model = classifier.fit(train_x1, train_y1)
y_pred2 = model.predict(test_x1)
add_col2 = [] # 第二组的新增特征值
for i in range(len(y_pred2)):
    tmp = y_pred2[i]
    add_col2.append(tmp)

"### 第三组重划分 219~327 数据为测试集"

test_x1 = func_divide_datasetX(218,327)
test_y1 = func_divide_datasetY(218,327)
train_x1 = np.concatenate((func_divide_datasetX(0,218),func_divide_datasetX(327,544)),axis=0)
train_y1 = np.concatenate((func_divide_datasetY(0,218),func_divide_datasetY(327,544)),axis=0)
model = classifier.fit(train_x1, train_y1)
y_pred3 = model.predict(test_x1)
add_col3 = [] # 第三组的新增特征值
for i in range(len(y_pred3)):
    tmp = y_pred3[i]
    add_col3.append(tmp)

"### 第四组重划分 328~436 数据为测试集"

test_x1 = func_divide_datasetX(327,436)
test_y1 = func_divide_datasetY(327,436)
train_x1 = np.concatenate((func_divide_datasetX(0,327),func_divide_datasetX(436,544)),axis=0)
train_y1 = np.concatenate((func_divide_datasetY(0,327),func_divide_datasetY(436,544)),axis=0)
model = classifier.fit(train_x1, train_y1)
y_pred4 = model.predict(test_x1)
add_col4 = [] # 第四组的新增特征值
for i in range(len(y_pred4)):
    tmp = y_pred4[i]
    add_col4.append(tmp)

"### 第五组重划分 437~544 数据为测试集"

test_x1 = func_divide_datasetX(436,544)
test_y1 = func_divide_datasetY(436,544)
train_x1 = func_divide_datasetX(0,436)
train_y1 = func_divide_datasetY(0,436)
model = classifier.fit(train_x1, train_y1)
y_pred5 = model.predict(test_x1)
add_col5 = [] # 第四组的新增特征值
for i in range(len(y_pred5)):
    tmp = y_pred5[i]
    add_col5.append(tmp)


"# 合并RF模型产生的预测数据 作为新的特征值"

add_col_rf = add_col1 + add_col2 + add_col3 + add_col4 + add_col5
add_col_RF = []
for i in range(len(add_col_rf)):
    if add_col_rf[i] == 1:
        tmp = [1]
        add_col_RF.append(tmp)
    if add_col_rf[i] == 2:
        tmp = [2]
        add_col_RF.append(tmp)
    if add_col_rf[i] == 3:
        tmp = [3]
        add_col_RF.append(tmp)
    if add_col_rf[i] == 4:
        tmp = [4]
        add_col_RF.append(tmp)
    if add_col_rf[i] == 5:
        tmp = [5]
        add_col_RF.append(tmp)
    if add_col_rf[i] == 6:
        tmp = [6]
        add_col_RF.append(tmp)
    if add_col_rf[i] == 7:
        tmp = [7]
        add_col_RF.append(tmp)
    

#add_col_RF = preprocessing.StandardScaler().fit_transform(add_col_RF)  
#scaler =  preprocessing.StandardScaler() # 然后生成一个标准化对象
#add_col_RF = scaler.fit_transform(add_col_RF)  #然后对数据进行转换  
add_col_RF = np.array(add_col_RF)

##############################################################################

"####### DT 重新划分数据集 赋予新的特征值"

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth = 6,max_leaf_nodes = 16,
                                    criterion = 'entropy',min_samples_split= 5, min_samples_leaf = 4,random_state = 0)


"### 第一组重划分 1~109 数据为测试集"
test_x1 = func_divide_datasetX(0,109)
test_y1 = func_divide_datasetY(0,109)
train_x1 = func_divide_datasetX(109,544)
train_y1 = func_divide_datasetY(109,544)
model = clf.fit(train_x1, train_y1)
y_pred1 = model.predict(test_x1)
add_col1 = [] # 第一组的新增特征值
for i in range(len(y_pred1)):
    tmp = y_pred1[i]
    add_col1.append(tmp)

"### 第二组重划分 110~218 数据为测试集"

test_x1 = func_divide_datasetX(109,218)
test_y1 = func_divide_datasetY(109,218)
train_x1 = np.concatenate((func_divide_datasetX(0,109),func_divide_datasetX(218,544)),axis=0)
train_y1 = np.concatenate((func_divide_datasetY(0,109),func_divide_datasetY(218,544)),axis=0)
model = clf.fit(train_x1, train_y1)
y_pred2 = model.predict(test_x1)
add_col2 = [] # 第二组的新增特征值
for i in range(len(y_pred2)):
    tmp = y_pred2[i]
    add_col2.append(tmp)

"### 第三组重划分 219~327 数据为测试集"

test_x1 = func_divide_datasetX(218,327)
test_y1 = func_divide_datasetY(218,327)
train_x1 = np.concatenate((func_divide_datasetX(0,218),func_divide_datasetX(327,544)),axis=0)
train_y1 = np.concatenate((func_divide_datasetY(0,218),func_divide_datasetY(327,544)),axis=0)
model = clf.fit(train_x1, train_y1)
y_pred3 = model.predict(test_x1)
add_col3 = [] # 第三组的新增特征值
for i in range(len(y_pred3)):
    tmp = y_pred3[i]
    add_col3.append(tmp)

"### 第四组重划分 328~436 数据为测试集"

test_x1 = func_divide_datasetX(327,436)
test_y1 = func_divide_datasetY(327,436)
train_x1 = np.concatenate((func_divide_datasetX(0,327),func_divide_datasetX(436,544)),axis=0)
train_y1 = np.concatenate((func_divide_datasetY(0,327),func_divide_datasetY(436,544)),axis=0)
model = clf.fit(train_x1, train_y1)
y_pred4 = model.predict(test_x1)
add_col4 = [] # 第四组的新增特征值
for i in range(len(y_pred4)):
    tmp = y_pred4[i]
    add_col4.append(tmp)

"### 第五组重划分 437~544 数据为测试集"

test_x1 = func_divide_datasetX(436,544)
test_y1 = func_divide_datasetY(436,544)
train_x1 = func_divide_datasetX(0,436)
train_y1 = func_divide_datasetY(0,436)
model = clf.fit(train_x1, train_y1)
y_pred5 = model.predict(test_x1)
add_col5 = [] # 第四组的新增特征值
for i in range(len(y_pred5)):
    tmp = y_pred5[i]
    add_col5.append(tmp)


"# 合并DT模型产生的预测数据 作为新的特征值"

add_col_dt = add_col1 + add_col2 + add_col3 + add_col4 + add_col5
add_col_DT = []
for i in range(len(add_col_dt)):
    if add_col_dt[i] == 1:
        tmp = [1]
        add_col_DT.append(tmp)
    if add_col_dt[i] == 2:
        tmp = [2]
        add_col_DT.append(tmp)
    if add_col_dt[i] == 3:
        tmp = [3]
        add_col_DT.append(tmp)
    if add_col_dt[i] == 4:
        tmp = [4]
        add_col_DT.append(tmp)
    if add_col_dt[i] == 5:
        tmp = [5]
        add_col_DT.append(tmp)
    if add_col_dt[i] == 6:
        tmp = [6]
        add_col_DT.append(tmp)
    if add_col_dt[i] == 7:
        tmp = [7]
        add_col_DT.append(tmp)
 
#add_col_DT = preprocessing.StandardScaler().fit_transform(add_col_DT)  
#scaler =  preprocessing.StandardScaler() # 然后生成一个标准化对象
#add_col_DT = scaler.fit_transform(add_col_DT)  #然后对数据进行转换
add_col_DT = np.array(add_col_DT)

"##############################################################################"

"####### LR 重新划分数据集 赋予新的特征值"

from sklearn.linear_model import LogisticRegression
logisticregression = LogisticRegression(C = 0.5,multi_class='multinomial')

"### 第一组重划分 1~109 数据为测试集"
test_x1 = func_divide_datasetX(0,109)
test_y1 = func_divide_datasetY(0,109)
train_x1 = func_divide_datasetX(109,544)
train_y1 = func_divide_datasetY(109,544)
model = logisticregression.fit(train_x1, train_y1)
y_pred1 = model.predict(test_x1)
add_col1 = [] # 第一组的新增特征值
for i in range(len(y_pred1)):
    tmp = y_pred1[i]
    add_col1.append(tmp)

"### 第二组重划分 110~218 数据为测试集"

test_x1 = func_divide_datasetX(109,218)
test_y1 = func_divide_datasetY(109,218)
train_x1 = np.concatenate((func_divide_datasetX(0,109),func_divide_datasetX(218,544)),axis=0)
train_y1 = np.concatenate((func_divide_datasetY(0,109),func_divide_datasetY(218,544)),axis=0)
model = logisticregression.fit(train_x1, train_y1)
y_pred2 = model.predict(test_x1)
add_col2 = [] # 第二组的新增特征值
for i in range(len(y_pred2)):
    tmp = y_pred2[i]
    add_col2.append(tmp)

"### 第三组重划分 219~327 数据为测试集"

test_x1 = func_divide_datasetX(218,327)
test_y1 = func_divide_datasetY(218,327)
train_x1 = np.concatenate((func_divide_datasetX(0,218),func_divide_datasetX(327,544)),axis=0)
train_y1 = np.concatenate((func_divide_datasetY(0,218),func_divide_datasetY(327,544)),axis=0)
model = logisticregression.fit(train_x1, train_y1)
y_pred3 = model.predict(test_x1)
add_col3 = [] # 第三组的新增特征值
for i in range(len(y_pred3)):
    tmp = y_pred3[i]
    add_col3.append(tmp)

"### 第四组重划分 328~436 数据为测试集"

test_x1 = func_divide_datasetX(327,436)
test_y1 = func_divide_datasetY(327,436)
train_x1 = np.concatenate((func_divide_datasetX(0,327),func_divide_datasetX(436,544)),axis=0)
train_y1 = np.concatenate((func_divide_datasetY(0,327),func_divide_datasetY(436,544)),axis=0)
model = logisticregression.fit(train_x1, train_y1)
y_pred4 = model.predict(test_x1)
add_col4 = [] # 第四组的新增特征值
for i in range(len(y_pred4)):
    tmp = y_pred4[i]
    add_col4.append(tmp)

"### 第五组重划分 437~544 数据为测试集"

test_x1 = func_divide_datasetX(436,544)
test_y1 = func_divide_datasetY(436,544)
train_x1 = func_divide_datasetX(0,436)
train_y1 = func_divide_datasetY(0,436)
model = logisticregression.fit(train_x1, train_y1)
y_pred5 = model.predict(test_x1)
add_col5 = [] # 第四组的新增特征值
for i in range(len(y_pred5)):
    tmp = y_pred5[i]
    add_col5.append(tmp)


"# 合并LR模型产生的预测数据 作为新的特征值"

add_col_lr = add_col1 + add_col2 + add_col3 + add_col4 + add_col5
add_col_LR = []
for i in range(len(add_col_lr)):
    if add_col_lr[i] == 1:
        tmp = [1]
        add_col_LR.append(tmp)
    if add_col_lr[i] == 2:
        tmp = [2]
        add_col_LR.append(tmp)
    if add_col_lr[i] == 3:
        tmp = [3]
        add_col_LR.append(tmp)
    if add_col_lr[i] == 4:
        tmp = [4]
        add_col_LR.append(tmp)
    if add_col_lr[i] == 5:
        tmp = [5]
        add_col_LR.append(tmp)
    if add_col_lr[i] == 6:
        tmp = [6]
        add_col_LR.append(tmp)
    if add_col_lr[i] == 7:
        tmp = [7]
        add_col_LR.append(tmp)


#scaler =  preprocessing.StandardScaler() # 然后生成一个标准化对象
#add_col_LR = scaler.fit_transform(add_col_LR)  #然后对数据进行转换
add_col_LR = np.array(add_col_LR)

df = pd.DataFrame({
    'KNN': add_col_KNN.flatten(),
    'GBT': add_col_GB.flatten(),
    'NB': add_col_NB.flatten(),
    'SVC': add_col_SVC.flatten(),
    'RF': add_col_RF.flatten(),
    'DT': add_col_DT.flatten(),
    'MLR': add_col_LR.flatten(),
})

# 保存DataFrame到CSV文件
df.to_csv('output.csv', index=False)


from sklearn.datasets import load_iris
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
X, y = load_iris(return_X_y=True)
kernel = 1.0 * RBF(1.0)
gpc = GaussianProcessClassifier(kernel=kernel,
        random_state=0).fit(X_train, y_train)


#print(gpc.score(X_train, y_train))

#print(gpc.predict_proba(X_train[:2,:]))


"##############################################################################"

"# 新增的8个特征值加入训练集中 (根据单一模型预测准确度,主观地赋予了一定权重) "



X_train = np.append(X_train, add_col_LR*1.5, axis=1)
X_train = np.append(X_train, add_col_KNN*0.9, axis=1)
X_train = np.append(X_train, add_col_DT*0.8, axis=1)
X_train = np.append(X_train, add_col_RF*1.0, axis=1)
X_train = np.append(X_train, add_col_SVC*1.0,axis=1)
X_train = np.append(X_train, add_col_GB*1.35, axis=1)
X_train = np.append(X_train, add_col_NB*0.25, axis=1)

""
#X_train = mm.fit_transform(X_train) 


#print(X_train)




new_y = []
for i in range(len(y_train)):
    if y_train[i] == 1: #A
        tmp = [0]
        new_y.append(tmp)
    if y_train[i] == 2: #B+
        tmp = [1]
        new_y.append(tmp)
    if y_train[i] == 3: #B
        tmp = [2]
        new_y.append(tmp)
    if y_train[i] == 4: #B-
        tmp = [3]
        new_y.append(tmp)
    if y_train[i] == 5: #C
        tmp = [4]
        new_y.append(tmp)
    if y_train[i] == 6: #D 
        tmp = [5]
        new_y.append(tmp)
    if y_train[i] == 7: #F
        tmp = [6]
        new_y.append(tmp)
new_y = np.array(new_y)
y_train = new_y

X_train = preprocessing.normalize(X_train, norm='l2') # 全局数据正则化处理



"BP神经网络 训练新数据"



import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import  DataLoader
import torch.nn.functional as F #使用functional中的ReLu激活函数
import torch.optim as optim

import torch.nn.functional as Fun

#设置超参数
lr= 0.001 #学习率
epochs = 10000 #训练轮数
n_feature = 12 #输入特征
n_hidden = 20 #隐层节点数
n_output = 7 #输出(类别)

 
#设置训练集数据80%，测试集20%
x_train0,x_test0,y_train_BP,y_test= train_test_split (X_train,y_train,test_size = 0.2, random_state = 0)

min_max_scaler = preprocessing.MinMaxScaler()
x_train = min_max_scaler.fit_transform(x_train0)
x_test = min_max_scaler.fit_transform(x_test0)
 
#1.将数据类型转换为tensor方便pytorch使用
x_train = torch.FloatTensor(x_train0)
y_train_BP = torch.LongTensor(y_train_BP)
x_test = torch.FloatTensor(x_test0)
y_test = torch.LongTensor(y_test)

#2.定义BP神经网络
class BPNetModel(torch.nn.Module):
    def __init__(self,n_feature,n_hidden,n_output):
        super(BPNetModel, self).__init__()
        self.hiddden = torch.nn.Linear(n_feature,n_hidden)#定义隐层网络
        self.out = torch.nn.Linear(n_hidden,n_output)#定义输出层网络
    def forward(self,x):
        x = Fun.relu(self.hiddden(x)) #隐层激活函数采用relu()函数
        out = Fun.softmax(self.out(x),dim=1) #输出层采用softmax函数
        return out
#3.定义优化器和损失函数
net = BPNetModel(n_feature = n_feature,n_hidden = n_hidden,n_output = n_output) #调用网络
optimizer = torch.optim.Adam(net.parameters(),lr=lr) #使用Adam优化器，并设置学习率
loss_fun = torch.nn.MultiLabelSoftMarginLoss() #对于多分类一般使用交叉熵损失函数

#4.训练数据
loss_steps = np.zeros(epochs) #构造一个array([ 0., 0., 0., 0., 0.])里面有epochs个0
accuracy_steps = np.zeros(epochs)
Max = 0
for epoch in range(epochs):
    y_pred = net(x_train) #前向传播
    loss = loss_fun(y_pred,y_train_BP)#预测值和真实值对比
    optimizer.zero_grad() #梯度清零
    loss.backward() #反向传播
    optimizer.step() #更新梯度
    loss_steps[epoch] = loss.item()#保存loss
    running_loss = loss.item()
    #print(f"第{epoch}次训练，loss={running_loss}".format(epoch,running_loss))
    with torch.no_grad(): #下面是没有梯度的计算,主要是测试集使用，不需要再计算梯度了
        y_pred = net(x_test)
        
        correct = (torch.argmax(y_pred,dim = 1) == y_test).type(torch.FloatTensor)
        accuracy_steps[epoch] = correct.mean()
        Max = max(Max,accuracy_steps[epoch])
       # print("测试的预测准确率", accuracy_steps[epoch])
 

#print(Max) bp神经网络的最大准确率 


"BP神经网络"



"xgboost模型调参"


from sklearn.feature_selection import SelectFromModel


import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn import metrics
 
model = XGBClassifier(learning_rate=0.01,
                      n_estimators=500,           # 树的个数-10棵树建立xgboost
                      max_depth=4,               # 树的深度
                      min_child_weight = 0.05,      # 叶子节点最小权重
                      gamma=0.,                  # 惩罚项中叶子结点个数前的参数
                      subsample=1,               # 所有样本建立决策树
                      colsample_btree=0,         # 所有特征建立决策树
                      scale_pos_weight=1,        # 解决样本个数不平衡的问题
                      random_state=1,           # 随机数
                      slient = 0,
                      booster='gbtree',
                      objective='binary:logistic',
                      )


X_train = preprocessing.normalize(X_train, norm='l2') # 全局数据正则化处理

X_embedded = SelectFromModel(model,threshold=0.0067).fit_transform(X_train,y_train)
#print(X_embedded)



X_train_new, X_test_new, y_train_new, y_test_new = train_test_split (X_embedded,y_train,test_size = 0.2, random_state = 0)

model.fit(X_train_new,y_train_new)
y_pred = model.predict(X_test_new)


"xgboost输出结果与真实值的等级映射"

"定义成绩等级的映射函数 "

def grade_mapping(grade):
    if grade in [0,1]:
        return 0
    elif grade in [2,3]:
        return 2
    elif grade in [4,5]:
        return 4
    else :
        return 6



"对预测结果和真实结果应用等级映射"

mapped_predictions = [grade_mapping(prediction) for prediction in y_pred]
mapped_labels = [grade_mapping(label) for label in y_test_new]

print("prediction length:{}".format(len(mapped_predictions)))


"计算准确率"

correct = 0

for i in range(len(mapped_predictions)):
    if mapped_predictions[i] == mapped_labels[i]:
        correct = correct + 1
        
"输出等级映射后的xgboost模型准确率"

print("xgboost accuracy: ",correct/len(y_test_new)) 

