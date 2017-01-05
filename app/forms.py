from app import app
from flask import request, render_template
from app.analyze.bankinator import Bankinator


@app.route("/")
def predict_mark():
    return render_template("predict_mark.html")


@app.route("/predict", methods=['GET'])
def predictor():
    nostradamus = Bankinator()
    return "I see.. Your mark is " + str(nostradamus.predict(request.args.get('text', '')))
