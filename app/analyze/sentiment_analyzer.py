__author__ = 'izakharkin'

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib


class SentimentAnalyzer():
    def __init__(self):
        self.model = joblib.load('./ml_models/XGBClassifierModel.pkl')
        self.vectorizer = joblib.load("./vectorizers/TfidfVectorizer.pkl")
        self.classes_dict = {0: "awful", 1: "very negative", 2: "negative",
                             3: "satisfied", 4: "good", 5: "great", -1: "prediction error"}

    def predict_text(self, text):
        try:
            vectorized = self.vectorizer.transform([text])
            return self.model.predict(vectorized)[0], \
                   self.model.predict_proba(vectorized)[0].max()
        except:
            print('prediction error')
            return -1, 0.8

    def predict_list(self, list_of_texts):
        try:
            vectorized = self.vectorizer.transform(list_of_texts)
            return self.model.predict(vectorized), \
                   self.model.predict_proba(vectorized)
        except:
            print('prediction error')
            return None

    @staticmethod
    def get_probability_words(probability):
        if probability < 0.55:
            return "neutral or uncertain"
        if probability < 0.7:
            return "probably"
        if probability > 0.95:
            return "certain"
        else:
            return ""

    def get_prediction_message(self, text):
        prediction = self.predict_text(text)
        class_prediction = prediction[0]
        prediction_probability = prediction[1]
        return self.get_probability_words(prediction_probability) + " " + self.classes_dict[class_prediction]

    def predict(self, input_data):
        response = [input_data]
        response = self.vectorizer.transform(response)
        return self.model.predict(response)
