import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./fair_trail2.csv')

individual = df.iloc[:, 0]
group = df.iloc[:, 1]
accuracy = df.iloc[:, 2]

plt.figure(figsize=(10, 6))

plt.plot(df.index, individual, label='individual_loss', color='blue')

plt.plot(df.index, group, label='group_loss', color='green')

plt.plot(df.index, accuracy, label='accuracy', color='red')

plt.xlabel('Iterations')
plt.ylabel('Value')
plt.title('Loss&Accuracy')
plt.legend()

plt.show()
