import mlflow.pyfunc

# connect to same MLflow DB
mlflow.set_tracking_uri("sqlite:///C:/Users/acer/OneDrive/Desktop/mlops/mlflow.db")

# load production model
model = mlflow.pyfunc.load_model("models:/Churn_Model@production")

print("Production model loaded ✅")
