__author__ = 'izakharkin'

from sklearn.externals import joblib


class SentimentAnalyzer():
    def __init__(self):
        self.model = joblib.load('./classifiers/BestClassifier.pkl')
        self.vectorizer = joblib.load("./vectorizers/BestVectorizer.pkl")
        vocabulary = joblib.load('./vectorizers/best_vectorizer_vocabulary.pkl')
        self.vectorizer.vocabulary_ = vocabulary
        self.classes_dict = {0: "awful", 1: "very negative", 2: "negative",
                             3: "satisfied", 4: "good", 5: "great", -1: "prediction error"}
        self.can_predict_proba = 'predict_proba' in dir(self.model)

    def predict_text(self, text):
        try:
            vectorized = self.vectorizer.transform([text])
            return self.model.predict(vectorized)[0], \
                   (self.model.predict_proba(vectorized)[0].max() if self.can_predict_proba else '')
        except:
            print('prediction error')
            return -1, 0.8

    def predict_list(self, list_of_texts):
        try:
            vectorized = self.vectorizer.transform(list_of_texts)
            return self.model.predict(vectorized), \
                   (self.model.predict_proba(vectorized) if self.can_predict_proba else '')
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
            return ''

    def get_prediction_message(self, text):
        prediction = self.predict_text(text)
        class_prediction = prediction[0]
        prediction_probability = prediction[1]
        words = (self.get_probability_words(prediction_probability) if self.can_predict_proba \
                     else 'I see.. The mark is')
        return words + ' ' + str(class_prediction) + ' (' + self.classes_dict[class_prediction] + ')'

    def predict(self, input_data):
        response = [input_data]
        response = self.vectorizer.transform(response)
        return self.model.predict(response)
