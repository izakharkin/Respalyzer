import pandas as pd
import numpy as np

# Reading and showing the data
data = pd.read_csv("responses_dataset.csv")
print(data.head())
print(data.shape)
print(data.describe().T)

X = data['description']
y = data['mark']

print(pd.value_counts(y))

# doing Oversampling to balance the classes
np.random.seed(0)
X_added = X
prev_y = y
for cur_mark in range(2, 6):
    oversample_size = np.abs(np.sum(y == 1) - np.sum(y == cur_mark))
    indices_to_add = np.random.randint(np.sum(y == cur_mark), size=oversample_size)

    X_to_add = X[indices_to_add]
    X_added = np.hstack((X_added, X_to_add))
    y_added = np.concatenate((prev_y, np.ones(oversample_size) * cur_mark))
    prev_y = y_added

print(pd.value_counts(prev_y))