import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Balanced():
    def __init__(self) -> None:
        self.trail = pd.read_csv("./fair_trail2.csv")
        self.individual_trail = self.trail["individual"]
        self.accuracy_trail = self.trail["accuracy"]
        self.alpha = 0.5
        self.beta = (1 - self.alpha) / max(self.individual_trail)
        self.n = len(self.accuracy_trail)
        self.results = []
        
    def traget(self, accuracy, fairness_loss):
        return self.alpha * accuracy - self.beta * fairness_loss
    
    def analysis_trail(self):
        for i in range(self.n):
            self.results.append(self.traget(self.accuracy_trail[i], self.individual_trail[i]))
        df = pd.DataFrame(self.results, columns=['Target_function'])
        df.to_csv('balanced_trail.csv', index=False)

    def show_trail(self):
        df = pd.read_csv('balanced_trail.csv')
        y = df['Target_function']
        x = range(self.n)
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, marker='o', linestyle='-')
        plt.xlabel('Iteration')
        plt.ylabel('Target Function')
        plt.grid(True)
        plt.show()

    def show_relation(self):
        df = pd.read_csv('fair_trail2.csv')
        y = df['individual']
        x = df['accuracy']

        plt.figure(figsize=(10, 6))
        plt.plot(x, y, marker='o', linestyle='-', color='b')
        plt.xlabel('accuracy')
        plt.ylabel('fairness_loss')
        plt.grid(True)
        plt.show()

b = Balanced()
#b.analysis_trail()
#b.show_trail()
b.show_relation()
