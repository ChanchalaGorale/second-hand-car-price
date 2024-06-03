from flask import Flask, render_template, request, redirect
import os
import logging

logging.basicConfig(level=logging.DEBUG)

import pickle

import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve(strict=True).parent

#with open(f"{BASE_DIR}/notebook/path/to/price/pipeline.pkl", "rb") as f:
    #model = pickle.load(f)

model = pickle.load(open(f"{BASE_DIR}/notebook/path/to/price/pipeline.pkl", 'rb'))

port = int(os.environ.get('PORT', 5000))

app = Flask(__name__)


@app.route("/",  methods=["GET", "POST"])
def home():

    result=""

    if request.form:

        years = request.form['years'] 
        km = request.form['km']
        rating = request.form['rating']
        condition = request.form['condition']
        economy = request.form['economy']
        speed = request.form['topSpeed']
        hp = request.form['hp']
        torque = request.form['torque']

        arr = np.array([years, km, rating, condition, economy, speed, hp, torque], dtype=np.float64)
        

        if len(arr) == 8 :
            result = model.predict([arr.reshape(1, 8)])[0][0]
           

    return  render_template("home.html",result=result )


if __name__ =="__main__":
    app.run(host='0.0.0.0', port=port, debug=True)