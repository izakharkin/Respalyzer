from app import app
from flask import request, render_template
from app.analyze.sentiment_analyzer import Bankinator


@app.route("/act")
def predict_mark():
    return render_template("predict_mark.html")


@app.route("/predict", methods=['GET'])
def predictor():
    nostradamus = Bankinator()
    return "I see.. Your mark is " + str(nostradamus.predict(request.args.get('text', '')))
