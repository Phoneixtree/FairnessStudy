import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Measurement import Measurement
#from sklearn.metrics import multilabel_confusion_matrix

class ConfusionMatrix:
    def __init__(self,info):
        self.classes=['A','B+','B','B-','C','D','F']
        gamma = np.array([
        [1.0000, 0.9125, 0.8250, 0.7500, 0.5875, 0.3750, 0.0000],
        [0.9125, 1.0000, 0.9125, 0.8375, 0.6625, 0.4625, 0.1250],
        [0.8250, 0.9125, 1.0000, 0.9250, 0.7375, 0.5500, 0.2125],
        [0.7500, 0.8375, 0.9250, 1.0000, 0.8375, 0.6250, 0.2875],
        [0.5875, 0.6625, 0.7375, 0.8375, 1.0000, 0.7875, 0.4500],
        [0.3750, 0.4625, 0.5500, 0.6250, 0.7875, 1.0000, 0.6625],
        [0.0000, 0.1250, 0.2125, 0.2875, 0.4500, 0.6625, 1.0000]])
        self.levels=len(self.classes)
        self.pred=np.array(info['pred'])
        self.true=np.array(info['true'])
        self.comb=np.column_stack((self.true,self.pred))
        self.matrix=np.zeros((self.levels,self.levels),dtype=int)
        self.confusion_matrix(self.comb)
        m=Measurement({'matrix':self.matrix,'gamma':gamma})
        #self.matrix_show()
        #self.calculate_metrics()


    def confusion_matrix(self,combination):
        for c in combination:
            tmpTrue=self.classes.index(c[0])
            tmpPred=self.classes.index(c[1])
            self.matrix[tmpTrue][tmpPred]+=1

    def calculate_metrics(self):
        precision = []
        recall = []
        for i in range(self.levels):
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

            

