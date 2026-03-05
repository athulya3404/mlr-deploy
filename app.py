from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open("models/mlr_model.pkl","rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    exp = float(request.form["experience"])
    edu = float(request.form["education"])
    hours = float(request.form["hours"])
    score = float(request.form["score"])

    features = np.array([[exp, edu, hours, score]])

    prediction = model.predict(features)

    return render_template("index.html",
            prediction_text="Predicted Salary : {}".format(round(prediction[0],2)))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003)