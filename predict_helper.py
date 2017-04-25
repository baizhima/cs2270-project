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
	selected_attr_list = f.read().split(',')

with open('selected_model_type.txt', 'rb') as f:
	model_type = int(f.read().split(',')[0])

def load_predict(x):
	#select features for x test
	# x = df[df['id'] == id & (df['timestamp'] == time_stamp)].loc[:, selected_attr_list]
	#do the feature selection
	feature = x.loc[:, selected_attr_list].values

	#test using the model
	if model_type == 3:
		X_test = xgb.DMatrix(feature)
		model = xgb.Booster({'nthread':4})
		model.load_model("lag1.model") 
		return model.predict(X_test)

	model = joblib.load_model("lag1.model")
	return model.predict(feature)