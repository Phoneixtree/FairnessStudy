import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
#from Measurement import Measurement
#from sklearn.metrics import multilabel_confusion_matrix

class ConfusionMatrix:
    def __init__(self,args):
        with open(os.path.dirname(os.path.abspath(__file__))+"/MatrixSetting_OULAD.json","r") as json_file:
            info=json.load(json_file)
        self.classes=info["levels"]
        self.level_length=len(self.classes)
        #df=pd.read_csv(info["comparison_trail"])
        #self.prediction=args["origin_prediction"]
        self.cali_prediction=args["calibrated_prediction"]
        self.actual=args["actual"]
        #self.pred=np.array(info['pred'])
        #self.true=np.array(info['true'])
        self.comb=np.column_stack((self.actual,self.cali_prediction))

        self.initial_matrix=np.zeros((self.level_length,self.level_length),dtype=int)
        self.relation_matrix = np.array(info["relation_matrix"])
        self.confusion_matrix(self.comb)

        self.matrix=np.dot(self.initial_matrix,self.relation_matrix)
        #m=Measurement({'matrix':self.matrix,'gamma':gamma})
        #self.matrix_show()
        #self.calculate_metrics()

    def confusion_matrix(self,combination):
        for c in combination:
            tmpTrue=self.classes.index(c[0])
            tmpPred=self.classes.index(c[1])
            self.initial_matrix[tmpTrue][tmpPred]+=1

    def calculate_metrics(self):
        precision = []
        recall = []
        for i in range(self.level_length):
            TP = self.matrix[i, i]
            FP = self.matrix[:, i].sum() - TP
            FN = self.matrix[i, :].sum() - TP
            if (TP + FP) > 0:
                precision.append(TP / (TP + FP))
            else:
                precision.append(0.0)
            if (TP + FN) > 0:
                recall.append(TP / (TP + FN))
            else:
                recall.append(0.0)

        self.metrics_df = pd.DataFrame({'Precision': precision, 'Recall': recall}, index=self.classes)
        print("\nPrecision and Recall for each class:")
        print(self.metrics_df)
        self.visualize_metrics()

    def visualize_metrics(self):
        plt.figure(figsize=(8, 5))
        sns.heatmap(self.metrics_df, annot=True, cmap='Blues', fmt='.2f', linewidths=.5)
        plt.title('Precision and Recall for each class')
        plt.xlabel('Metrics')
        plt.ylabel('Classes')
        plt.show()

    def matrix_show(self):
        df = pd.DataFrame(self.matrix, index=self.classes, columns=self.classes).astype(int)

        # Calculate row sums and column sums
        row_sums = df.sum(axis=1)
        col_sums = df.sum(axis=0)

        # Add totals to the DataFrame
        df['Total'] = row_sums
        df.loc['Total'] = col_sums.append(pd.Series(row_sums.sum(), index=['Total']))

        #print(df)

        # Visualize the confusion matrix using seaborn
        plt.figure(figsize=(10, 7))
        ax=sns.heatmap(df, annot=True, fmt='d', cmap='Greys', cbar=False, 
                    linecolor='black',linewidths=0.5,
                    xticklabels=df.columns, yticklabels=df.index)  # Use DataFrame's own labels
        plt.xlabel('Predict')
        plt.ylabel('True')
        plt.title('Multi-class Confusion Matrix')

        for _, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(0.5)

        plt.show()

            

