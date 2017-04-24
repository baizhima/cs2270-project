def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pandas as pd
import numpy as np

from sklearn.externals import joblib
from sklearn import cross_validation
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from xgboost import XGBRegressor
import time
import pickle


#k value of k-fold cv
fold_num = 5
max_feature_num = 16
#path to 5000 samples for feature selection
sample_file_path = "./shift/lag5000.csv"


def read_file(file_name):
	'''
		read the preprocessed lag data from disk
		return the data frame

	'''
	# print "read all data from files"
	df = pd.read_csv(file_name)
	return df


def split_into_train_test(df):

	'''
	split the data into training set and test set
	first 4 / 5 is for training, the rest 1 / 5 is for test

	extract all features among attrubites, 
	return data frame for X_train, X_test, Y_train, Y_test

	'''
	training_set = df.iloc[0: len(df) * 4 / 5]
	print "training records ", len(training_set)
	test_set = df.iloc[len(df) * 4 / 5 :]
	print "test records ", len(test_set)
	
	#prepare training feautres and labels, include id and timestamp
	df_X_train = training_set.iloc[:,0: 98]
	df_Y_train = training_set['y1']


	df_X_test = test_set.iloc[:,0: 98]
	df_Y_test = test_set['y1']

	#return df for each one
	return df_X_train, df_Y_train, df_X_test, df_Y_test




def get_r_value(r2):
	'''
	compute r values (coefficient of determination) from R-squared
	
	'''
	return np.sign(r2) * np.sqrt(np.abs(r2))




def eval_r(Y_predict, dtrain):

	'''
	customazed evaluation metircs for xgboost corss validation 
	Input: Y_predict, np array
	 	   dtrain, XGB DMatrix
	'''

	r2_val = r2_score(dtrain.get_label(), Y_predict)
	return 'r', get_r_value(r2_val)



def feature_selection(X, Y, max_feature_num):
	'''

	This function performs feature-selection
	Input: X -- df_X_train
		   Y -- df_Y_train
	 	   max_feature_num

	Output: selected_attr_list, selected_model_type, corresponding cv r_value

	Hardcoded ranked ranked attribute

	'''

	#ranking list from corrlation matrix, descending order, totally 59

	all_attr_list = ['y', 'technical_30', 
					'technical_30_lag1', 
					'technical_27_lag1', 
					'technical_27', 
					'technical_19_lag1', 
					'technical_19', 
					'technical_35_lag1',
					 'technical_35', 
					 'technical_11_lag1','technical_36_lag1','technical_20' ,'y_lag1','technical_36','technical_2_lag1','fundamental_53_lag1', 
					'fundamental_53','technical_11','technical_2','fundamental_18' ,'fundamental_18_lag1' ,'technical_43' ,'technical_43_lag1' , 'technical_6_lag1' , 'technical_6'  ,'technical_0'  ,'technical_13_lag1' ,
					'technical_14' ,'technical_14_lag1' ,'technical_21_lag1' ,'technical_41' ,'fundamental_42_lag1', 
					'technical_42_lag1' ,'fundamental_42', 'technical_41_lag1','fundamental_48_lag1' ,'derived_0_lag1' ,'fundamental_48' ,'technical_33' ,        
                	'technical_39_lag1' ,   
                'fundamental_41',        
                 'derived_0',            
                 'technical_21',          
                 'fundamental_45_lag1',   
                 'fundamental_41_lag1' ,   
                 'fundamental_45',        
                 'technical_33_lag1',     
                 'technical_42',        
                 'fundamental_19' ,        
                 'fundamental_19_lag1',
                 'fundamental_33',       
                 'fundamental_33_lag1', 
                 'fundamental_36_lag1',  
                 'fundamental_36',     
                 'fundamental_21',     
                 'fundamental_7_lag1', 
                 'fundamental_7']

	print "attribute len " , len(all_attr_list)

    #selected feature to be returned
	selected_attr_list = []
	#selected type
	model_type = [-1]
	#performance of selected type
	max_r_val = [float("-inf")]

	chunk_size = 4
	#for each chunk
	for i in range(0, len(all_attr_list) / chunk_size):
		print "chunk" , i, '-----------------------------------'

		#special start and end of the current chunk
		start = i * chunk_size
		end= (i + 1) * chunk_size

		print "start ", start, " end ", end
		print "start wrapper"

		#within a chunk, dp wrapper method
		wrapper(selected_attr_list, model_type , max_r_val,  X, Y, all_attr_list, start, end)

		print "end wrapper"
		print "selected_attr_list ", selected_attr_list 
		print "model type selected ", model_type[0]
		print "r_val ", max_r_val[0]

		if len(selected_attr_list) >= max_feature_num:
			break
	print "final selected selected_attr_list ", selected_attr_list
	print "final current type ", model_type[0]
	print "final r value ", max_r_val[0]


	with open('selected_attr_list.txt','wb') as f:
	    f.write( ','.join(selected_attr_list))
	with open('selected_model_type.txt', 'wb') as f:
		f.write('%s'%model_type[0])
		# f.write(','.join(str(int(x)) for x in model_type))

	# return selected_attr_list, model_type[0], max_r_val[0]



def wrapper(selected_attr_list, model_type, max_r_val, X, Y, all_attr_list, start, end):
	'''
	This function performs wrapper method to select features with k-fold cross validation
	e.g [a, b, c, d]
	iter 1: [a               0.6       checked
	           b             0.5
	              c          0.4
	                d]       0.3
	iter 2: [ab
	          ac
	            ad]        checked
	iter3: not better than ad, return and keep track of the [current selected attr list, model_type, r_val]

	then move on to the next chunk
	'''

	#4 iterations max
	for iter_num in range(0, 4):

		#initialize array
		print "iteration ", iter_num
		iter_acc = np.empty(4)     #record cv r val for each combination
		iter_acc.fill(float("-inf"))   #init to -inf

		iter_type = np.empty(4)    #record cv model type for each combination, select from [0,1,2,3]
		iter_type.fill(-1)		#init to -1

		print "current selected attribute list ", selected_attr_list
		print "current model type ", model_type[0]
		print "current r value ", max_r_val[0]

		#snapshot of current selected
		temp = selected_attr_list
		for i in range(start, end):
			print "i = " , i

			#if selected, pass
			if all_attr_list[i] in temp:
				print "continue"
				continue


			temp.append(all_attr_list[i])
			print "cur combo",  temp

			#do cross validation on 4 candidate models, and return mean r val for each model type
			cross_val_mean_scores = time_series_cv(X, Y, temp)

			#keep track of max r val and its index
			max_value = np.amax(cross_val_mean_scores)
			max_index = np.argmax(cross_val_mean_scores)

			print "cv value ", max_value

			iter_acc[i - start] = max_value
			iter_type[i - start] = max_index

			temp.pop()

		print "iter_acc ", iter_acc
		print "iter_type ", iter_type

		if np.max(iter_acc) <= max_r_val[0]:
			#not better than the biggest r_val,  return from wrapper methond, move to the next chunk
			return

		#otherwise, append attr to selected
		selected_attr_list.append(all_attr_list[np.argmax(iter_acc) + start])
		#update model type
		model_type[0] = int(iter_type[np.argmax(iter_acc)])
		#update r val
		max_r_val[0] = np.max(iter_acc)




def time_series_cv(X, Y, attr_list):


	'''
	K-fold cross validation for time series data
	e.g k = 5, steps are as followed
	1. Split train set into 5 consecutive time folds.
	2. for each type of model, 
			 [0]: Linear Regressor
			 [1]: SVR
			 [2]: Random Forest
			 [3]: XGBoost

		train on fold 1, test on 2
		train on fold 1,2 test on 3
		train on fold 1,2,3  test on 4
		train on 1,2, 3,4  test on 5
	3. compute the mean of 4 accuracies.

	Input: X  -- sample features
		   Y  -- sample true values
		   fold_num --- k
		   attr_list  --- a list of string of selected attribute
	'''



	#chunk size
	fold_size = X.shape[0] / fold_num
	#init acc matrix, size : 4 * fold_num - 1
	acc = np.zeros((4, fold_num - 1))

	#travese all the types, each row represents k-fold cv for the corresponding type
	'''
		 0: lr
		 1: svm
		 2: rf
		 3: xgboost
	'''
	for row in range(0, 4):
		for col in range(2, fold_num + 1):
			end_index = fold_size * col
			index = int(end_index * (col - 1) * 1. / col)
			# print index
			X_train = X[:index]
			Y_train = Y[:index]
			X_test = X[index + 1 : end_index]
			Y_test = Y[index + 1 : end_index]
			_, acc[row, col - 2] =  regression(X_train, Y_train, X_test, Y_test, attr_list, row)

	print acc
	# print acc mean for each model type, axis = y
	print np.mean(acc, axis = 1);
	return np.mean(acc, axis = 1);


#run the training process
def regression(X_train, Y_train, X_test, Y_test, attr_list, type):

	#select attributes, transforms to np array
	X_train = X_train.loc[:, attr_list].values
	# print "Number of attribute ", X_train.shape[1]
	X_test = X_test.loc[:, attr_list].values
	Y_train = Y_train.values
	Y_test = Y_test.values

	#initiate regressor for each type
	if type == 0:
		# print "linear regression"
		regressor = LinearRegression()
	elif type == 1:
		# print "SVR"
		regressor = SVR(kernel = 'rbf')
	elif type == 2:
		# print "Random Forest"
		regressor = RandomForestRegressor(n_estimators=100, 
										  oob_score = True, 
										  n_jobs = -1,
										  random_state = 50, 
										  max_features = "auto", 
										  min_samples_leaf = 50)

	else:
	# 	regressor = XGBRegressor(max_depth=3, 
	# 			   learning_rate=0.05, 
	# 			   n_estimators=100, silent=True, 
	# 			   objective='reg:linear',
	# 			   min_child_weight=Y_train.size/2000,
	# 			   colsample_bytree=1, 
	# 			   base_score=Y_train.mean())

	# 	# print "self-tuned xgboost"

	# 	#set parameters

		params_xgb = {'objective'    : 'reg:linear',
          			  'tree_method'      : 'hist',
          			  'grow_policy'      : 'depthwise',
         			  'eta'              : 0.05,   
                      'subsample'        : 0.6,
          			  'max_depth'        : 10,    
          			  'min_child_weight' : Y_train.size/2000, 
          			  'colsample_bytree' : 1, 
        			  'base_score'       : Y_train.mean(),
          			  'silent'           : True,
		}

		#number of boosting iterations
		n_round = 16  
		#tranform into dmatrix object
		xg_train = xgb.DMatrix(X_train, label=Y_train)

		#train the xgboost model
		model = xgb.train(params_xgb, xg_train, num_boost_round = n_round, verbose_eval = False)
		#predict on test data 
		Y_predict = model.predict(xgb.DMatrix(X_test))
		#return test r value
		return model, get_r_value(r2_score(Y_test, Y_predict))


	regressor.fit(X_train, Y_train)
	Y_predict = regressor.predict(X_test)
	return regressor, get_r_value(r2_score(Y_test, Y_predict))



def gradient_boosting(X_train, Y_train):

	'''
		Compare the performance of two xgboost models
		namely, default params and self-tuned params

	'''

	print ""
	print "xgboost"

	print "Gradient Boosting Regressor  --DEFAULT PARAMS"
	xgb_clf = XGBRegressor(max_depth=3, 
					   learning_rate=0.05, 
					   n_estimators=100, silent=True, 
					   objective='reg:linear',
					   min_child_weight=Y_train.size/2000,
					   colsample_bytree=1, 
					   base_score=Y_train.mean())

	# scores = cross_validation.cross_val_score(xgb_clf, X_train, Y_train, cv=10, scoring='r2')
	print scores
	r2_mean = scores.mean()
	r2_std = scores.std()
	print "xgbvalidation r2 mean: "+ str(r2_mean)
	print "rf validation r2 std: "+ str(r2_std)
	print "rf validation r value " + str(get_r_value(r2_mean))


	xgb_clf.fit(X_train, Y_train)
	joblib.dump(xgb_clf, 'xgb_clf.model')


	print ""
	print "Gradient Boosting -- SELF-TUNED PARAMS"
	params_xgb = {'objective'    : 'reg:linear',
              'tree_method'      : 'hist',
              'grow_policy'      : 'depthwise',
              'eta'              : 0.05,   #0.3 //learning rate , 0.05
              'subsample'        : 0.6,
              'max_depth'        : 10,    #6
              'min_child_weight' : Y_train.size/2000,  #1
              'colsample_bytree' : 1, 
              'base_score'       : Y_train.mean(),
              'silent'           : True,
	}

	n_round = 16  #number of boosting iterations
	xg_train = xgb.DMatrix(X_train, label=Y_train)
	model = xgb.train(params_xgb, xg_train, num_boost_round = n_round, verbose_eval = False)

	# # print "cross validation"
	xgb.cv(params_xgb, xg_train, n_round, nfold=10, feval=eval_r, seed = 0, callbacks=[xgb.callback.print_evaluation()])
	print "Saving model"
	model.save_model('train_lag1.model')



def main():

	df = read_file("./shift/lag5000.csv")

	df_X_train, df_Y_train, df_X_test, df_Y_test = split_into_train_test(read_file(sample_file_path))
	start_time = time.time()
	feature_selection(df_X_train, df_Y_train, max_feature_num)
	print "Time taken : ", time.time() - start_time



if __name__ == '__main__':
	main()

