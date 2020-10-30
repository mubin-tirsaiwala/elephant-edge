#!/usr/bin/env python
# coding: utf-8

# In[6]:

import sys
sys.path.append('/home/mubin/myenv/lib/python3.6/site-packages/flask_cors')
from flask import Flask, request, redirect, url_for, flash, jsonify, render_template
from flask_cors import CORS, cross_origin
import numpy as np
import json
import tensorflow as tf
import geojson

x_dynacc_feeding = np.random.normal(0.028, 0.010, 200)
y_dynacc_feeding = np.random.normal(0.023, 0.008, 200)
z_dynacc_feeding = np.random.normal(0.018, 0.005, 200)

x_dynacc_bathing = np.random.normal(0.073, 0.042, 200)
y_dynacc_bathing = np.random.normal(0.061, 0.027, 200)
z_dynacc_bathing = np.random.normal(0.039, 0.022, 200)

x_dynacc_walking = np.random.normal(0.091, 0.027, 200)
y_dynacc_walking = np.random.normal(0.084, 0.025, 200)
z_dynacc_walking = np.random.normal(0.053, 0.013, 200)

x_dynacc_swaying = np.random.normal(0.130, 0.012, 200)
y_dynacc_swaying = np.random.normal(0.089, 0.008, 200)
z_dynacc_swaying = np.random.normal(0.070, 0.008, 200)

x_data = np.hstack((x_dynacc_feeding, x_dynacc_bathing, x_dynacc_walking, x_dynacc_swaying))
y_data = np.hstack((y_dynacc_feeding, y_dynacc_bathing, y_dynacc_walking, y_dynacc_swaying))
z_data = np.hstack((z_dynacc_feeding, z_dynacc_bathing, z_dynacc_walking, z_dynacc_swaying))


app = Flask(__name__)
CORS(app)

@app.route('/')
@cross_origin()
def index():
    return render_template("index.html")

@app.route('/allelephants/', methods = ['GET'])
def allelephants():
    with open('geojson_files/allelephants.geojson') as f:
        return(geojson.load(f))

@app.route('/DUDU/', methods = ['GET'])
def DUDU():
    with open('geojson_files/DUDU.geojson') as f:
        return(geojson.load(f))

@app.route('/FREDERICO/', methods = ['GET'])
def FREDERICO():
    with open('geojson_files/FREDERICO.geojson') as f:
        return(geojson.load(f))

@app.route('/HECTOR/', methods = ['GET'])
def HECTOR():
    with open('geojson_files/HECTOR.geojson') as f:
        return(geojson.load(f))

@app.route('/HENRIQUE/', methods = ['GET'])
def HENRIQUE():
    with open('geojson_files/HENRIQUE.geojson') as f:
        return(geojson.load(f))

@app.route('/JOAQUIM/', methods = ['GET'])
def JOAQUIM():
    with open('geojson_files/JOAQUIM.geojson') as f:
        return(geojson.load(f))

@app.route('/LUCAS/', methods = ['GET'])
def LUCAS():
    with open('geojson_files/LUCAS.geojson') as f:
        return(geojson.load(f))

@app.route('/MANOEL/', methods = ['GET'])
def MANOEL():
    with open('geojson_files/MANOEL.geojson') as f:
        return(geojson.load(f))

@app.route('/MOGLI/', methods = ['GET'])
def MOGLI():
    with open('geojson_files/MOGLI.geojson') as f:
        return(geojson.load(f))

@app.route('/NANDO/', methods = ['GET'])
def NANDO():
    with open('geojson_files/NANDO.geojson') as f:
        return(geojson.load(f))

@app.route('/PEDROCA/', methods = ['GET'])
def PEDROCA():
    with open('geojson_files/PEDROCA.geojson') as f:
        return(geojson.load(f))

@app.route('/api/', methods=['GET', 'POST'])
def makecalc():
    import time
    t = int(str(int(time.time()))[-1])
    if request.method == 'GET':
        x = x_data[t*50 : t*50+200]
        y = y_data[t*50 : t*50+200]
        z = z_data[t*50 : t*50+200]
    
    data = list(np.ravel(np.hstack((np.expand_dims(x,-1), np.expand_dims(y,-1), np.expand_dims(z,-1)))))

    interpreter.set_tensor(input_details[0]['index'], np.expand_dims(np.asarray(data).astype(np.float32), 0))
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    classes = ['bathing', 'feeding', 'swaying', 'walking']
    prediction = classes[np.argmax(output)]
    return jsonify(x.tolist(),
                   y.tolist(),
                   z.tolist(),
                   prediction
                    )

if __name__ == '__main__':
    modelfile = 'ei-elephant-edge-10hz-nn-classifier-tensorflow-lite-float32-model.lite'
    interpreter = tf.lite.Interpreter(modelfile)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    app.run(debug=True, host='0.0.0.0')

