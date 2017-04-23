from app import app
from flask import request, render_template
from app.src.sentiment_analyzer import SentimentAnalyzer


@app.route("/act")
def predict_mark():
    return render_template("predict_mark.html")


@app.route("/predict", methods=['GET'])
def predictor():
    respalizer = SentimentAnalyzer()
    return respalizer.get_prediction_message(request.args.get('text', ''))

# <dd><textarea name=text rows=5 cols=40>{{ text }}</textarea>