from flask import Blueprint, render_template, flash, redirect, url_for, request
from flask_bootstrap import __version__ as FLASK_BOOTSTRAP_VERSION
from flask_nav.elements import Navbar, View, Subgroup, Link, Text, Separator

from app.forms import TextInputForm
from app.nav import nav

from app.src.sentiment_analyzer import SentimentAnalyzer

frontend = Blueprint('frontend', __name__)

nav.register_element('frontend_top', Navbar(
    View('Respalizer', '.index'),
    View('Home', '.index'),
    View('Sentiment form', '.sentiment_form'),
    View('Debug-Info', 'debug.debug_root'),
    Subgroup(
        'Documentation',
        Link('About', 'https://github.com/izaharkin/Respalizer/blob/master/README.md'),
        Link('Help', 'https://github.com/izaharkin/Respalizer/wiki'),
        Link('Technologies', 'https://github.com/izaharkin/Respalizer/wiki/Used-technologies'), ),
    Text('Using Flask-Bootstrap {}'.format(FLASK_BOOTSTRAP_VERSION)), ))


@frontend.route('/')
def index():
    return render_template('index.html')


@frontend.route('/sentiment', methods=['GET'])
def sentiment_form():
    form = TextInputForm()
    return render_template('sentiment_form.html', form=form)


@frontend.route("/predict", methods=['GET'])
def predict():
    sentiment_text = request.args.get('textarea', '')
    respalyzer = SentimentAnalyzer()
    prediction_message = respalyzer.get_prediction_message(sentiment_text)
    return render_template("prediction.html", form=TextInputForm(),
                           sentiment_text=sentiment_text,
                           prediction_message=prediction_message)
