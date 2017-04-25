def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn

import pandas as pd
import numpy as np

from sklearn.externals import joblib
import xgboost as xgb
import time
import pickle
import read as rd

with open('selected_attr_list.txt', 'rb') as f:
	selected_attr_list_1 = f.read().split(',')

with open('selected_model_type.txt', 'rb') as f:
	model_type_1 = int(f.read().split(',')[0])

with open('selected_attr_list_10.txt', 'rb') as f:
	selected_attr_list_10 = f.read().split(',')

with open('selected_model_type_10.txt', 'rb') as f:
	model_type_10 = int(f.read().split(',')[0])

def load_predict(x):
	#select features for x test
	# x = df[df['id'] == id & (df['timestamp'] == time_stamp)].loc[:, selected_attr_list]
	#do the feature selection
	res = []
	
	feature_1 = x.loc[:, selected_attr_list_1].values
	#test using the model
	if model_type_1 == 3:
		X_test_1 = xgb.DMatrix(feature_1)
		model_1 = xgb.Booster({'nthread':4})
		model_1.load_model("lag1.model") 
		res.append(model_1.predict(X_test_1))
	else:
		model_1 = joblib.load_model("lag1.model")
		res.append(model_1.predict(feature_1))


	feature_10 = x.loc[:, selected_attr_list_10].values
	#test using the model
	if model_type_10 == 3:
		X_test_10 = xgb.DMatrix(feature_10)
		model_10 = xgb.Booster({'nthread':4})
		model_10.load_model("lag1.model") 
		res.append(model_10.predict(X_test_10))
	else:
		model_10 = joblib.load_model("lag1_y10.model")
		res.append(model_10.predict(feature_10))

	return res