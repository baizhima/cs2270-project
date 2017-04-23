import pandas as pd
import numpy as np

from sklearn.externals import joblib
from sklearn import cross_validation
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

# from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from xgboost import XGBRegressor
import time


def read_file(file_name):
	print "read all data from files"
	df = pd.read_csv(file_name)
	return df


def split_into_train_test(df):
	#split into train and test
	training_set = df.iloc[0: len(df) * 4 / 5]
	print "training records " + str(len(training_set))
	test_set = df.iloc[len(df) * 4 / 5 :]
	print "test records " + str(len(test_set))
	
	# training_features = training_set.iloc[:, 2:52]
	# first_part = training_set.iloc[:, 2: 50]
	# second_part = training_set.iloc[:, 51:]
	# training_features = pd.concat([training_set.iloc[:, 2: 50], training_set.iloc[:, 51:]], axis = 1)

	#prepare training feautres and labels
	X_train = training_set.iloc[:,2: 98].values
	Y_train = training_set['y1'].values

	# training_labels = np.asarray(training_set['y1'], dtype="float")
	# test_features = pd.concat([test_set.iloc[:, 2: 50], test_set.iloc[:, 51:]], axis = 1)

	X_test = test_set.iloc[:,2: 98].values
	# test_labels = np.asarray(test_set['y1'], dtype="float")
	Y_test = test_set['y1'].values

	return X_train, Y_train, X_test, Y_test


'''	
	This function train the data with following classifiers
	1. Linear Regression
	2. SVR: kernel-> {Gaussian, Linear, Poly with degree 2}
	3. Random Forest

'''
def cross_val(X_train, Y_train):

	print "linear regression starts..."
	lr = LinearRegression()
	# lr.fit(X_train, Y_train)
	scores = cross_validation.cross_val_score(lr, X_train, Y_train, cv=10, scoring='r2')
	print scores
	r2_mean = scores.mean()
	r2_std = scores.std()
	print "LR validation r2 mean: "+ str(r2_mean)
	print "LR validation r2 std: "+ str(r2_std)
	print "LR validation r value " + str(get_r_value(r2_mean))
	#save model here
	joblib.dump(lr, 'linear_regression.model') 

	print " "
	print "--------------------------------"
	print "SVR with"
	print "Gaussian Kernel:"
	svr_rbf = SVR(kernel='rbf')
	# svr_rbf.fit(X_train, Y_train)
	scores = cross_validation.cross_val_score(svr_rbf, X_train, Y_train, cv=10, scoring='r2')
	print scores
	r2_mean = scores.mean()
	r2_std = scores.std()
	print "svr_rbf validation r2 mean: "+ str(r2_mean)
	print "svr_rbf validation r2 std: "+ str(r2_std)
	print "svr_rbf validation r value " + str(get_r_value(r2_mean))
	#save model here
	joblib.dump(svr_rbf, 'svr_rbf.model') 


	print ""
	print "Linear Kernel:"
	svr_lin = SVR(kernel='linear')
	# svr_lin.fit(X_train, Y_train)
	scores = cross_validation.cross_val_score(svr_lin, X_train, Y_train, cv=10, scoring='r2')
	print scores
	r2_mean = scores.mean()
	r2_std = scores.std()
	print "svr_rbf validation r2 mean: "+ str(r2_mean)
	print "svr_rbf validation r2 std: "+ str(r2_std)
	print "svr_rbf validation r value " + str(get_r_value(r2_mean))
	#save model here
	joblib.dump(svr_lin, 'svr_lin.model') 

	print ""
	print "Poly kernel with degree 2:"
	svr_poly = SVR(kernel='poly', degree=4)
	# svr_poly.fit(X_train, Y_train)
	scores = cross_validation.cross_val_score(svr_poly, X_train, Y_train, cv=10, scoring='r2')
	print scores
	r2_mean = scores.mean()
	r2_std = scores.std()
	print "svr_rbf validation r2 mean: "+ str(r2_mean)
	print "svr_rbf validation r2 std: "+ str(r2_std)
	print "svr_rbf validation r value " + str(get_r_value(r2_mean))
	#save model here
	joblib.dump(svr_poly, 'svr_lin.model') 


	print ""
	print "----------------------------------------"
	print "RandomForestRegressor"

	#default
	rf = RandomForestRegressor(n_estimators=100, oob_score = True, n_jobs = -1,random_state = 50, max_features = "auto", min_samples_leaf = 50)
	start_time = time.time()
	rf.fit(X_train, Y_train)
	elapsed_time = time.time() - start_time
	scores = cross_validation.cross_val_score(rf, X_train, Y_train, cv=10, scoring='r2')
	print scores
	r2_mean = scores.mean()
	r2_std = scores.std()
	print "rf validation r2 mean: "+ str(r2_mean)
	print "rf validation r2 std: "+ str(r2_std)
	print "rf validation r value " + str(get_r_value(r2_mean))
	#save model here
	joblib.dump(rf, 'random_forest.model') 
	print elapsed_time


def get_r_value(r2):
	return np.sign(r2) * np.sqrt(np.abs(r2))


def eval_r(Y_predict, dtrain):
	r2_val = r2_score(dtrain.get_label(), Y_predict)
	return 'r', get_r_value(r2_val)


#5 fold validation

# fold 1 : training [1], test [2]
# fold 2 : training [1 2], test [3]
# fold 3 : training [1 2 3], test [4]
# fold 4 : training [1 2 3 4], test [5]
# fold 5 : training [1 2 3 4 5], test [6]
def self_corss_validation(X_train, Y_train):






def gradient_boosting(X_train, Y_train):
	print ""
	print "xgboost"

	print "Gradient Boosting Regressor  --DEFAULT PARAMS"
	xgb_clf = XGBRegressor(max_depth=3, 
					   learning_rate=0.1, 
					   n_estimators=100, silent=True, 
					   objective='reg:linear',
					   min_child_weight=Y_train.size/2000,
					   colsample_bytree=1, 
					   base_score=Y_train.mean())

	scores = cross_validation.cross_val_score(xgb_clf, X_train, Y_train, cv=10, scoring='r2')
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
	model = xgb.train(params_xgb, xg_train, num_boost_round = n_round, verbose_eval = True)

	# # print "cross validation"
	xgb.cv(params_xgb, xg_train, n_round, nfold=10, feval=eval_r, seed = 0, callbacks=[xgb.callback.print_evaluation()])
	print "Saving model"
	model.save_model('train_lag1.model')


def load_test(file_name, X_test, Y_test, is_xgboost_self):
	#for other classifiers
	if is_xgboost_self:
		X_test = xgb.DMatrix(X_test)
		model = xgb.Booster({'nthread':4}) #init model
		model.load_model(file_name) 
	else:
		model = joblib.load(file_name)
	Y_predict= model.predict(X_test)
	r2 = r2_score(Y_test, Y_predict)
	print "test r values is " + str(get_r_value(r2))


def main():
	# df = read_file("./shift/preprocessed_data_no_na.csv")
	# df = read_file("./shift/line5000.csv")
	# df = read_file("./shift/lag5000.csv")
	df = read_file("./shift/train_lag1_new.csv")

	df_X_train, df_Y_train, df_X_test, df_Y_test = split_into_train_test(df)
	cross_val(df_X_train, df_Y_train)
	# gradient_boosting(df_X_train,df_Y_train)
	load_test('random_forest.model', df_X_test, df_Y_test, False)

	#load classifiers with sklearn
	# clf = joblib.load('filename.pkl')


	# model = xgb.Booster({'nthread':4}) #init model
	# model.load_model("train_lag1.model") 
	# test(model, test_features, test_labels)


if __name__ == '__main__':
	main()

