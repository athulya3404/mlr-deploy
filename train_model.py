import os
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

# read dataset with correct delimiter (file is tab-separated)
dataset = pd.read_csv("MLR.csv", sep="\t")

X = dataset[['Experience_Years','Education_Years','Monthly_Hours','Performance_Score']]
y = dataset['Salary']

model = LinearRegression()
model.fit(X, y)

# ensure output directory exists
os.makedirs("models", exist_ok=True)
pickle.dump(model, open("models/mlr_model.pkl", "wb"))

print("MLR model trained and saved")