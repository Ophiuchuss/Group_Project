from flask import Flask, render_template, request, session, Response
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pandas.io.json import json_normalize
import csv
import os
import base64
import json
import pickle
from werkzeug.utils import secure_filename
 
import cv2
import datetime
import dlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time 

#*** Backend operation
 
# WSGI Application
# Provide template folder name
# The default folder name should be "templates" else need to mention custom folder name
 
# Accepted image for to upload for object detection model
UPLOAD_FOLDER = os.path.join('staticFiles', 'uploads')
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
 
app = Flask(__name__, template_folder='templateFiles', static_folder='staticFiles')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
 
app.secret_key = 'You Will Never Guess'
 
# AlexNet
def analysis_physiognomy(uploaded_image_path):
    # Landmark detection 
    result = []
    
    # Loading image
    image = cv2.imread(uploaded_image_path)
 
    image = cv2.resize(image, (480, 640), interpolation = cv2.INTER_AREA)
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
 
    # define the SVM + HOG  face detector
    detector = dlib.get_frontal_face_detector()
    # define the Dlib face landmark detector
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    detected_boxes = detector(image_rgb)
    
    # iterate over the total number of faces detected
    for box in detected_boxes:
        shape = predictor(image_rgb, box)
        
        '''
        # process the detection boxes
        res_box = process_boxes(box)
        cv2.rectangle(image, (res_box[0], res_box[1]),
                    (res_box[2], res_box[3]), (0, 255, 0), 
                    2)
        '''
        # iterate over all the keypoints
        for i in range(68): 
            # draw the keypoints on the detected faces
            cv2.circle(image, (shape.part(i).x, shape.part(i).y), 
                       2, (0, 255, 0), -1)
    
            result.append([shape.part(i).x, shape.part(i).y])
            
    result = np.asarray(result)

    # Write output image (object detection output)
    output_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output_image.jpg')
    cv2.imwrite(output_image_path, image)

    # Load the saved model - use ONLY when inference (not to re-train again/call specific saved model)
    model = tf.keras.models.load_model("physiognomy_AlexNet_1D.h5")
    
    # Predict for 1D input
    X_test = result.reshape((1, 68, 2))
    
    y_pred = model.predict(X_test)

    return(output_image_path, y_pred[0].tolist())
 
@app.route('/')
def index():
    return render_template('index_upload_and_display_image.html')
 
@app.route('/',  methods=("POST", "GET"))
def uploadFile():
    if request.method == 'POST':
        uploaded_img = request.files['uploaded-file']
        img_filename = secure_filename(uploaded_img.filename)
        uploaded_img.save(os.path.join(app.config['UPLOAD_FOLDER'], img_filename))
 
        session['uploaded_img_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)
 
        return render_template('index_upload_and_display_image_page2.html')
 
@app.route('/show_image')
def displayImage():
    img_file_path = session.get('uploaded_img_file_path', None)
    return render_template('show_image.html', user_image = img_file_path)
 
@app.route('/analyse_physiognomy')
def analyseFace():
    uploaded_image_path = session.get('uploaded_img_file_path', None)
    output_image_path, analysis_res = analysis_physiognomy(uploaded_image_path)
    print(output_image_path)
    return render_template('show_analysis.html', user_image = output_image_path, user_physiognomy = analysis_res)
 
# flask clear browser cache (disable cache)
# Solve flask cache images issue
@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response
 
if __name__=='__main__':
    app.run(debug = True)