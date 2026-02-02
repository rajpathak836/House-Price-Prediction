import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def train_model(data_path):
    df = pd.read_csv(data_path)
    X = df.drop("SalePrice", axis=1)
    y = df["SalePrice"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("MSE:", mean_squared_error(y_test, predictions))
    print("R2 Score:", r2_score(y_test, predictions))
    joblib.dump(model, "model/linear_regression_model.pkl")

if __name__ == "__main__":
    train_model("data/processed/cleaned_housing.csv")
