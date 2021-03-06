from flask import Flask, render_template, send_from_directory, request, \
   redirect, url_for, flash, send_file

from werkzeug.utils import secure_filename
import os
import sys

import tensorflow as tf

from tensorflow import keras
import numpy as np

import os


from sklearn.externals import joblib
import time
import os
import numpy as np
import glob
import shutil


import tensorflow_hub as hub



application = app = Flask(__name__)
app.secret_key = "super secret key"

@app.route('/')
def index():
    information = []
    return render_template('index.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/info')
def info():
    return render_template('info.html')

@app.route('/upload', methods=['GET','POST'])
def upload_file():
   if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        
        result1 = predictType(file)
        flash('success')
        return render_template('results.html',result=result1)
   return render_template('upload.html', title='Upload Image')

def predictType(filename):

   from PIL import Image
   import numpy as np
   from skimage import transform
   def load(filename1):
      np_image = Image.open(filename1)
      np_image = np.array(np_image).astype('float32')/255
      np_image = transform.resize(np_image, (150, 150, 3))
      np_image = np.expand_dims(np_image, axis=0)
      return np_image

   image = load(filename)

   
   model = '1584317473.h5'

   reloaded = tf.keras.models.load_model(model, custom_objects={'KerasLayer': hub.KerasLayer})
   

   arr = reloaded.predict(image)


   return arr



if __name__ == '__main__' :
    #app.run(debug=True)
    app.run(host='0.0.0.0',port=80)