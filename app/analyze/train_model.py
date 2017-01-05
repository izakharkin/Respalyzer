from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import cross_val_score
from xgboost import XGBRegressor

import pandas as pd
import numpy as np

# Reading and showing the data
data = pd.read_csv("responses_dataset.csv")
print(data.head())
print(data.shape)
print(data.describe().T)

# feature
X = data['description']
print(X[0])
# target
y = data['mark']
print(y[0])

vectorizer = TfidfVectorizer(sublinear_tf=True)
X = vectorizer.fit_transform(X)
print(X.shape)

def rmse(x, y):
    return np.mean((x - y) ** 2) ** 0.5

# Сравним градиентный бустинг с константой, чтобы понять, насколько всё хорошо (или плохо)
const_reg = y.mean()
print(rmse(const_reg, y))

xgb_reg = XGBRegressor()
xgb_reg.fit(X, y)

# здесь надо сохранить модель в файл, и каждый раз подгружать модель
# модель из файла в классе и PROFIY
#
# xgb_score = cross_val_score(xgb_reg, X, y)
# print(xgb_score.mean())
# потом в predict() округлять до ближайшего целого, т.к. мы всё же предсказываем оценку
print(rmse(xgb_reg.predict(X), y))

my_response = ["ВТБ24"]
my_response = vectorizer.transform(my_response)
print(xgb_reg.predict(my_response))