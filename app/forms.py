from flask_wtf import Form

from wtforms.fields import StringField
from wtforms.widgets import TextArea
from wtforms.validators import DataRequired


class TextInputForm(Form):
    textarea = StringField(u'Enter your sentiment',
                           widget=TextArea(),
                           validators=[DataRequired()])
