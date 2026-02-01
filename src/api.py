from fastapi import FastAPI
import pandas as pd
import mlflow
import mlflow.pyfunc

# connect MLflow DB
mlflow.set_tracking_uri("sqlite:///C:/Users/acer/OneDrive/Desktop/mlops/mlflow.db")

app = FastAPI()

# load production model
model = mlflow.pyfunc.load_model("models:/Churn_Model@production")

@app.get("/")
def home():
    return {"message": "API running"}

@app.post("/predict")
def predict(data: dict):

    df = pd.DataFrame([data])

    df = pd.get_dummies(df)

    X_train_cols = pd.read_csv(
        "C:/Users/acer/OneDrive/Desktop/mlops/data/processed/X_train.csv"
    ).columns

    df = df.reindex(columns=X_train_cols, fill_value=0)

    pred = model.predict(df)

    return {"prediction": int(pred[0])}
