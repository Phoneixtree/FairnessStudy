import numpy as np


class Measurement:
    def __init__(self,info):
        self.matrix=info['matrix']
        self.gamma=info['gamma']
        t=self.calculate_metrics(self.gamma,self.matrix,0)
        
    def calculate_metrics(self, gamma, Pred, A):
        n = gamma.shape[0]
        N = n
        TP_A = sum(gamma[i, A] * Pred[i][A] for i in range(n)) / N
        sum1 = sum(gamma[i, i] * Pred[i][i] for i in range(n) if i != A)
        sum2 = sum(gamma[i, j] * Pred[i][j] for i in range(n) if i != A for j in range(n) if j != A)
        TN_A = (-sum1 + sum2) / N
        FP_A = sum(gamma[i, A] * Pred[i][A] for i in range(n) if i != A) / N
        FN_A = sum(gamma[A, i] * Pred[A][i] for i in range(n) if i != A) / N
        print("TP_A: {:.4f}, TN_A: {:.4f}, FP_A: {:.4f}, FN_A: {:.4f}".format(TP_A, TN_A, FP_A, FN_A))
        return TP_A, TN_A, FP_A, FN_A