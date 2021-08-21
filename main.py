import pandas as pd
import numpy as np
import pickle
from flask import Flask,render_template,request

app = Flask(__name__)
data = pd.read_csv('Cleaned_data.csv')
pipe4 = pickle.load(open('lr_model.pkl','rb'))


@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    return render_template('index.html',locations=locations)


@app.route('/predict',methods=['POST'])
def predict():
    location = request.form.get('location')
    bhk = float(request.form.get('bhk'))
    bath = float(request.form.get('bath'))
    sqft = request.form.get('sqft')

    # print(location,bhk,bath,sqft)
    input = pd.DataFrame( [[location,bath,sqft,bhk]] , columns=['location','bath','total_sqft_float','bhk'])
    prediction = pipe4.predict(input)[0]*100000

    return str(np.round(prediction,2))

if __name__ == "__main__":
    app.run(debug=True)