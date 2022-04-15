import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('finalmodel.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]

    features_name =['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                    'BMI', 'DiabetesPedigreeFunction', 'Age',]

    df = pd.DataFrame(features_value, columns=features_name)
    output = model.predict(df)

    if output == 1:
        out = "oops you have diabetes"
    else:
        out = "congrats no diabetes"

    return render_template('index.html', prediction_text='****{}'.format(out))


if __name__ == "__main__":
    app.run()