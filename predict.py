import joblib
import numpy as np

model = joblib.load("model/linear_regression_model.pkl")
sample_house = np.array([[7, 2000, 800, 3, 2]])
print("Predicted House Price:", model.predict(sample_house)[0])
