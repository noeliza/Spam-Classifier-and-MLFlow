# Spam-Classifier Using Optuna and MLFlow
![R](https://img.shields.io/badge/Python-3.9-blue)

This project intends to show how to create a spam classifier using Optuna for hypertuning and MLFlow for tracking. Three classifiers are hypertuned, **SVM, XGBoost and RandomForest**. Based on the objective score(F1 Score), the best model for each classifiers are registered in MLFlow. Not all trials are logged as the objective is to compare the three models based on their best hyperparameters. All the logged models were evaluated using the test dataset. The objective score, evaluation metrics and classification reports are logged in MLFlow for tracking.
