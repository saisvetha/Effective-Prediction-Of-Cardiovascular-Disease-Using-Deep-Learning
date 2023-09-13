from flask import Flask, request, jsonify, render_template
import os
import pandas as pd
import pickle
import mlxtend
import cv2
from keras.models import load_model
from keras.utils import load_img,img_to_array
import numpy as np
app = Flask(__name__)
location=r'C:\Users\saisv\Pictures\cardiovascular\cardiovascular\model.h5'
model = pickle.load(open(location, 'rb'))

model1=load_model(r"C:\Users\saisv\Pictures\cardiovascular\cardiovascular\model1.h5")

app.config['UPLOAD_FOLDER'] = '/uploads/'
@app.route('/')
def home():
	return render_template('home.html')

@app.route('/image',methods=['GET', 'POST'])
def image():

    if request.method == 'POST':  
        f = request.files['file']
        f.save(f.filename) 
        path=f.filename 
        img = load_img(path, target_size=(150, 150))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x / 255.

    # Predict the class of the test image
        pred = model1.predict(x)[0]
        if(pred=='0'):
            pred='abnormal_heartbeat'
        elif(pred=='1'):
            pred='History_of_MI'
        elif(pred=='2'):
            pred='Myocardial Disease'
        else:
            pred='Normal'
        return render_template("image.html", pred = pred)  

@app.route('/pred',methods=['GET', 'POST'])
def pred():
	return render_template('app.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    x=''
    A= request.form['age']
    B= request.form['sex']
    if B=='f':
        B=0
    else:
        B=1
    C= request.form['cp']
    D = request.form['trestbps']
    E = request.form['glucose']
    F = request.form['BMI']
    G = request.form['chol']
    H = request.form['fbs']
    I = request.form['restecg']
    J = request.form['thalach']
    K = request.form['exang']
    L = request.form['oldpeak']
    M = request.form['slope']
    N = request.form['ca']
    O = request.form['thal']

    input_variables = pd.DataFrame([[A,B,C,D,E,F,G,H,I,J,K,L,M,N,O]],columns=['age','sex' ,'cp','trestbps','glucose','BMI','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal'])
    prediction = model.predict(input_variables)[0]
    if(prediction==0):
        x='normal'
    elif(prediction==1):
        x='coronary heart disease'
    elif(prediction==2):
        x='stroke'
    elif(prediction==3):
        x='peripheral arterial disease'
    elif(prediction==4):
        x='aortic disease.'
    else:
        x=''

    return render_template('app.html',result=x)
if __name__ == '__main__':
   app.run()


