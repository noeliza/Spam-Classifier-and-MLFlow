import pandas as pd
import numpy as np
import string
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sentence_transformers import SentenceTransformer
from collections import Counter
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.decomposition import PCA
import time

import optuna
import torch
import sklearn
import seaborn as sns
from functools import partial
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from imblearn.combine import SMOTETomek
import os
import mlflow
from imblearn.pipeline import Pipeline
from datetime import datetime
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
optuna.logging.set_verbosity(optuna.logging.WARNING)

#check if gpu exist for xgb
if torch.cuda.device_count() > 0:
	tree_method = 'gpu_hist'
else: tree_method = 'hist'


class preprocessor():
	
	def __init__(self, input_data, text, label, inference = False):
		self.input_data = input_data.copy()
		self.preprocess_data = None
		self.text = text
		self.label = label
	
	def unique(self):        
		print(f'Init - No of rows: {self.input_data.shape[0]}')    
		missing = self.input_data.shape[0] - self.input_data.dropna(axis = 0).shape[0]
		print(f'   Count of rows with missing values: {missing}')
		if missing > 0:
			print(f'      Removing rows with missing values...')        
			self.preprocess_data = self.input_data.dropna(axis = 0)
		else: self.preprocess_data = self.input_data.copy()
		dup = self.preprocess_data.duplicated(subset = [self.text]).sum()
		print(f'   Count of rows with duplicate predictor: {dup}')
		if dup > 0:
			print(f'      Removing rows with duplicate predictor...')
			self.preprocess_data.drop_duplicates(subset=self.text,ignore_index=True, inplace = True)
			
		print(f'Final - No of rows: {self.preprocess_data.shape[0]}')
		
		
	def label_summary(self):
		print(f'\n---------- Label Summary ----------')
		# display(pd.DataFrame({'count': self.preprocess_data[self.label].value_counts(),
		#                         '%': round(self.preprocess_data[self.label].value_counts(1)*100, 2)}))
		print(pd.DataFrame({'count': self.preprocess_data[self.label].value_counts(),
								'%': round(self.preprocess_data[self.label].value_counts(1)*100, 2)}))       
	
	def split_data(self, random_state = 42, test_size = 0.2):    
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.preprocess_data[self.text], self.preprocess_data[self.label],
												  test_size=test_size, 
												  random_state=random_state, 
												  stratify=self.preprocess_data[self.label].values)

		dist_train = pd.DataFrame({'Count - Train': self.y_train.value_counts(),
										'% - Train': round(self.y_train.value_counts(1)*100, 2)})

		dist_test = pd.DataFrame({'Count - Test': self.y_test.value_counts(),
										'% - Test': round(self.y_test.value_counts(1)*100, 2)})

		print(f'Splitting the data to {int(100-test_size*100)}:{int(test_size*100)} Train-Test...')
		print(f'\n---------- Train and Test Label Distribution ----------')
		print(dist_train.merge(dist_test, how = 'left', left_index = True, right_index = True))
   



class custom_pipeline(): 

	def __init__(self, X_train, y_train, X_test, y_test, n_components, random_state = 101,
		average = "macro", tune_scoring = "f1_macro", champ_scoring = "test recall",
		n_jobs = -1, kfold_splits = 5, trials = 100, overundersampling = SMOTETomek):
		self.X_train = X_train.reset_index(drop = True)
		self.y_train = y_train.reset_index(drop = True)
		self.X_test = X_test.reset_index(drop = True)
		self.y_test = y_test.reset_index(drop = True)
		self.n_components = n_components
		self.random_state = random_state
		self.trials = trials
		self.average = average
		self.tune_scoring = tune_scoring
		self.champ_scoring = champ_scoring
		self.kfold_splits = kfold_splits
		self.n_jobs = n_jobs
		self.overundersampling = overundersampling
		self.model_names = ["SVM", "XGBoost", "RandomForest"]
		self.models = {
			"SVM" : SVC,
			"XGBoost" : XGBClassifier,
			"RandomForest" : RandomForestClassifier,
		}
		self.best_params = {}
		self.best_models = {}
		self.hyperparameters = {
			"SVM" : [
				{
					"parameter" : "kernel",
					"type" : "categorical",
					"values" : ['poly','sigmoid','rbf']
				},
				{
					"parameter" : "C",
					"type" : "float",
					"values" : {
						"min" : 1e-3,
						"max" : 1e2
					}
				},
				{
					"parameter" : "gamma",
					"type" : "float",
					"values" : {
						"min" : 1e-3,
						"max" : 1e2
					}
				},
				{
					"parameter" : "degree",
					"type" : "integer",
					"values" : {
						"min" : 1,
						"max" : 8
					}
				},
				{
					"parameter" : "class_weight",
					"type" : "constant",
					"values" : "balanced"
				},
				{
					"parameter" : "random_state",
					"type" : "constant",
					"values" : self.random_state
				},
			],
			"XGBoost" : [
				{
					"parameter" : "n_estimators",
					"type" : "integer",
					"values" : {
						"min" : 100,
						"max" : 500
					}
				},
				{
					"parameter" : "max_depth",
					"type" : "integer",
					"values" : {
						"min" : 1,
						"max" : 11
					}
				},
				{
					"parameter" : "learning_rate",
					"type" : "float",
					"values" : {
						"min" : 1e-3,
						"max" : 1
					}
				},
				{
					"parameter" : "gamma",
					"type" : "float",
					"values" : {
						"min" : 0,
						"max" : 12
					}
				},
				{
					"parameter" : "subsample",
					"type" : "float",
					"values" : {
						"min" : 0.5,
						"max" : 1
					}
				},
				{
					"parameter" : "colsample_bytree",
					"type" : "float",
					"values" : {
						"min" : 0.5,
						"max" : 1
					}
				},
				{
					"parameter" : "tree_method",
					"type" : "constant",
					"values" : tree_method
				},
				{
					"parameter" : "random_state",
					"type" : "constant",
					"values" : self.random_state
				},
			],
			"RandomForest" : [
				{
					"parameter" : "n_estimators",
					"type" : "integer",
					"values" : {
						"min" : 100,
						"max" : 500
					}
				},
				{
					"parameter" : "max_depth",
					"type" : "integer",
					"values" : {
						"min" : 2,
						"max" : 11
					}
				},
				{
					"parameter" : "max_features",
					"type" : "float",
					"values" : {
						"min" : 0.1,
						"max" : 1
					}
				},
				{
					"parameter" : "min_samples_split",
					"type" : "float",
					"values" : {
						"min" : 0.1,
						"max" : 1
					}
				},
				{
					"parameter" : "min_samples_leaf",
					"type" : "float",
					"values" : {
						"min" : 0.1,
						"max" : 0.5
					}
				},
				{
					"parameter" : "class_weight",
					"type" : "constant",
					"values" : "balanced"
				},
				{
					"parameter" : "random_state",
					"type" : "constant",
					"values" : self.random_state
				},
			]
		}

	def distilbert_encoder(self, item):
		# Transformer Model
		distilbert = SentenceTransformer('distilbert-base-cased-distilled-squad')

		return distilbert.encode(item, convert_to_tensor = False, show_progress_bar = False)
			
	def dimensionality_reduction(self):
		print('Standardizing...')
		self.scaler = StandardScaler()
		self.scaler.fit(self.X_train_distilbert)

		X_scaled = pd.DataFrame(self.scaler.transform(self.X_train_distilbert))
		
		print('Initializing PCA...')
		self.pca = PCA(random_state = self.random_state)
		self.pca.fit(X_scaled)

		print('Applying PCA...')
		self.X_train_pca = pd.DataFrame(self.pca.transform(X_scaled)).iloc[:, 0:self.n_components]


	def get_hyperparameters(self, trial, parms):
		hyperparams ={}
		for parm in parms:
			hyperparams[parm['parameter']] = parm['values']
			if parm['type'] == 'constant':
				hyperparams[parm['parameter']] = parm['values']
			elif parm['type'] == 'categorical':
				hyperparams[parm['parameter']] = trial.suggest_categorical(parm['parameter'],parm['values'])
			elif parm['type'] == 'uniform':
				# Deprecated. Can use suggest float for now 
				hyperparams[parm['parameter']] = trial.suggest_uniform(parm['parameter'],parm['values']['min'],parm['values']['max'])
			elif parm['type'] == 'integer':
				# can pass in a step or set log = false... these values can not be used at the same time 
				hyperparams[parm['parameter']] = trial.suggest_int(parm['parameter'],parm['values']['min'],parm['values']['max'])
			elif parm['type'] == 'float':
				# can pass a step and and log value if needed
				hyperparams[parm['parameter']] = trial.suggest_float(parm['parameter'],parm['values']['min'],parm['values']['max'])
			elif parm['type'] == 'loguniform':
				hyperparams[parm['parameter']] = trial.suggest_float(parm['parameter'],parm['values']['min'],parm['values']['max'], log=True)
		return hyperparams
	


	def objective(self, trial, clf, clf_name,**others):
		# setting up model and parameters
		hyperparams = self.get_hyperparameters(trial, self.hyperparameters[clf_name])
		model = clf(**hyperparams)
		
		# calculating objective score
		objective_score = self.cross_validate(model,self.X_train_pca, self.y_train, self.tune_scoring, n_jobs=1)
		
		return objective_score
	


	def run_study(self, clf, clf_name, direction):
		print(f"Hypertuning : {clf_name}")
		
		# hypertuning
		start = time.time()
		optimization_function = partial(self.objective, clf = clf, clf_name = clf_name)
		study = optuna.create_study(direction = direction)
		study.optimize(optimization_function, n_trials = self.trials, n_jobs = self.n_jobs, show_progress_bar=True)
		end = time.time()
		
		print(f'Tuning time: {round((end - start)/60, 2)} min')
		print(f'Best score: {study.best_value}')
		print(f'Best params:')
		for key, value in study.best_trial.params.items():
			print("  {}: {}".format(key, value))            
		print(f"---------------------------------------------")

		return study.best_trial.params, study.best_value
		


	def hypertuning_and_evaluation(self):
		labels = self.y_train.value_counts()
		if np.min(labels)/np.sum(labels)*100 < 15:
			print('Applying Overundersampling...')
			self.apply_overundersampling = True


		print(f"---------- Hypertuning and Evaluating Classifiers ----------")


		# start loging using mlflow -------------------------------------------
		run_name = datetime.now().strftime('%Y-%m-%d-%Hh-%Mm-%Ss')
		with mlflow.start_run(run_name = run_name):
			# hypertuning all models
			for clf_name in self.model_names:
				# hypertuning each model
				study_result = self.run_study(self.models[clf_name], clf_name, direction ="maximize")
				self.best_params[clf_name] = study_result[0]
				
				params = self.hyperparameters[clf_name]
				
				for val in params:
					if val['type'] == 'constant':
						parm = val['parameter']
						self.best_params[clf_name][parm] = val['values']
				
				# defining model with best parameters
				# best_model = self.models[clf_name](**self.best_params[clf_name])
				# best_model.fit(self.X_train_pca, self.y_train)

				# defining model with best parameters and creating a pipeline
				if self.apply_overundersampling == True:
					best_model = Pipeline(
						[
							('vectorizer', FunctionTransformer(self.distilbert_encoder)),
							('standardizer', StandardScaler()),
							('pca', PCA(n_components = self.n_components, random_state = self.random_state)),
							('overundersampling', self.overundersampling(random_state = self.random_state)),
							('classifier', self.models[clf_name](**self.best_params[clf_name]))
						]
					)
					best_model.fit(self.X_train, self.y_train)

				else:
					best_model = Pipeline(
						[
							('vectorizer', FunctionTransformer(self.distilbert_encoder)),
							('standardizer', StandardScaler()),
							('pca', PCA(n_components = self.n_components, random_state = self.random_state)),
							('classifier', self.models[clf_name](**self.best_params[clf_name]))
						]
					)
					best_model.fit(self.X_train, self.y_train)


				# Evaluate on test dataset
				y_pred = best_model.predict(self.X_test)
				# print(classification_report(self.y_test, y_pred))
				metrics = {'accuracy': accuracy_score(self.y_test, y_pred),
					'precision': precision_score(self.y_test, y_pred, average=self.average),
					'recall': recall_score(self.y_test, y_pred,average=self.average),
					'f1 score':f1_score(self.y_test, y_pred, average=self.average)
					}

				print(f'Evaluation metrics:')
				for key, value in metrics.items():
					print("  {}: {}".format(key, value))       


				# with mlflow.start_run(nested=True, run_name = 'train-test-result') as run:
				with mlflow.start_run(nested = True):
				
					# log model params
					mlflow.log_params(self.best_params[clf_name])

					# log training score
					mlflow.log_metric('train_score', study_result[1]) 

					# log evaluation metrics
					mlflow.log_metrics(metrics) 

					# log confusion matrix as artifact
					fig, ax = plt.subplots()
					sns.heatmap(confusion_matrix(self.y_test,y_pred), fmt = 'd',annot=True, cmap='Blues')
					ax.set_title('Confusion Matrix')
					ax.set_ylabel('True Label')
					ax.set_xlabel('Predicted Label')
					mlflow.log_figure(fig, 'fig/confusion_matrix.png')


					# log model
					mlflow.sklearn.log_model(best_model, clf_name,
											 registered_model_name = 'spam-classifier-' + clf_name)



	def cross_validate(self, model, X, y, metrics=None, n_jobs=-1):        
		# initialization
		results_summary = []
		results_list = []
		X.reset_index(drop = True, inplace = True)
		y.reset_index(drop = True, inplace = True)
		
		# defining stratified k-fold split
		skf = StratifiedKFold(n_splits = self.kfold_splits, random_state = self.random_state, shuffle=True)
		
		# parallelized k-fold cross validation
		if n_jobs == 1:
			for train_index, test_index in skf.split(X, y):
				results_list.append(self.model_train(model, X, y, train_index, test_index, metrics))
		else:
			results_list = Parallel(n_jobs = n_jobs)(
				delayed(self.model_train)(
					model, X, y, train_index, test_index, metrics
				)
				for train_index, test_index in skf.split(X, y)
			)
		
		# metric mean and standard dev
		results_mean = np.mean(results_list, axis = 0)
		results_std = np.std(results_list, axis = 0)

		return results_mean
	


	def model_train(self, model, X, y, train_index, test_index, metrics=None):
		# train-test split based on skf index
		X_train, X_test = X.values[train_index], X.values[test_index]
		y_train, y_test = y[train_index], y[test_index]
		
		# determine whether to use SMOTE or not
		if self.apply_overundersampling == True:
			ousampling = self.overundersampling(random_state=self.random_state)
			X_train, y_train = ousampling.fit_resample(X_train, y_train)
			
		# model training
		model.fit(X_train, y_train)

		objective_score = [sklearn.metrics.get_scorer(metrics)(model, X_test, y_test)]
		return objective_score

		
	def run(self):
		self.X_train_distilbert = self.distilbert_encoder(self.X_train)
		self.dimensionality_reduction()
		self.hypertuning_and_evaluation()



		