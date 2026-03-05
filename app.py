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

    years = float(request.form["years"])
    education = float(request.form["education"])

    features = np.array([[years, education]])

    prediction = model.predict(features)

    return render_template("index.html",
           prediction_text="Predicted Salary : ₹{:,.0f}".format(prediction[0]))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003)