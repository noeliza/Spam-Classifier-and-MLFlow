import mlflow
import pandas as pd
import mlflow.pyfunc


data = pd.read_csv('./data/inference/input.csv', header = None)\
    .iloc[:, 0].tolist()

model_name = 'spam-classifier-SVM'
stage = 'Production'

model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{model_name}/{stage}"
)

model.predict(data)

