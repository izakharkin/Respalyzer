from flask import render_template
from app import app


@app.route('/')
@app.route('/index')
def index():
    user = {'nickname': 'Ilya'}
    return render_template("base.html",
                           # title = 'Home',
                           user=user)
