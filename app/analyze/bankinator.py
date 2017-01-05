import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.feature_extraction.text import TfidfVectorizer

class Bankinator():
    def predict(self, input_data):
        data = pd.read_csv("../responses_dataset.csv")

        X = data['description']
        y = data['mark']

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

        vectorizer = TfidfVectorizer(sublinear_tf=True)
        X = vectorizer.fit_transform(X)
        #
        # def rmse(x, y):
        #     return np.mean((x - y) ** 2) ** 0.5
        #
        xgb_reg = XGBRegressor()
        xgb_reg.fit(X, y)

        my_response = [input_data]
        my_response = vectorizer.transform(my_response)

        return int(xgb_reg.predict(my_response).mean()) #округлить до ближайшего целого