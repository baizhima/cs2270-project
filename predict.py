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



data_file_path = './data/train_lag1_new.csv'
df = rd.read_file(data_file_path)
model_dict = {0: 'Linear Regression',
			  1: 'SVR',
			  2: 'Random Forest',
			  3: 'XGBoost'}

with open('selected_attr_list.txt', 'rb') as f:
	selected_attr_list = f.read().split(',')

with open('selected_model_type.txt', 'rb') as f:
	model_type = int(f.read().split(',')[0])


def train_model():

	'''
	This function takes the best plan attribute, train on the whole training data
	save the model to the disk

	'''
	print "start training model_________________"
	
	df_X_train, df_Y_train, df_X_test, df_Y_test = rd.split_into_train_test(df)
	model, r_val = rd.regression(df_X_train, df_Y_train, df_X_test, df_Y_test,selected_attr_list, model_type)
	print "Test r_val : ", r_val
	if model_type == 3:
		#load xgboost model
		model.save_model("lag1.model")
	else:
		joblib.dump(model, "lag1.model")
	print "saving mode, model_type ", model_dict[model_type]



# x is df [id, timestamp, feature, feature_lag1]
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


def main():
	train_model()


if __name__ == '__main__':
	main()



