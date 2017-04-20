from math import log

feature_endpoints = {
'derived_0': (-25000, 5000),
'fundamental_0':(-2.5,1.5),
'fundamental_18':(-1200,200),
'fundamental_19':(-200,1200),
'fundamental_21':(-2,4),
'fundamental_33':(-1000000,6000000),
'fundamental_36':(-50000,350000),
'fundamental_41':(-3e8,1e8),
'fundamental_42':(-2e8,0.5e8),
'fundamental_45':(-10000,60000),
'fundamental_48':(-3000,3000),
'fundamental_53':(-700,100),
'fundamental_59':(-0.5,2),
'fundamental_7':(-100000,600000),
'technical_0':(-1,0),
'technical_11':(-2,0),
'technical_12':(-1,0),
'technical_13':(0,0.008),
'technical_14':(-2,0),
'technical_16':(-1,1),
'technical_17':(-2,0),
'technical_18':(-1,0),
'technical_19':(-10,70),
'technical_2':(-2,0),
'technical_20':(0.00,0.07),
'technical_21':(-200,1200),
'technical_22':(-0.6,0.6),
'technical_24':(-0.4,0.5),
'technical_27':(-5,25),
'technical_29':(-2,0),
'technical_3':(-0.3,0.4),
'technical_30':(0,0.08),
'technical_32':(-1.0,0.0),
'technical_33':(-1.0,1.5),
'technical_34':(-0.6,0.6),
'technical_35':(-2,16),
'technical_36':(-10,50),
'technical_37':(-1.0,0.0),
'technical_38':(-1.0,0.0),
'technical_39':(-1.0,0.0),
'technical_40':(-1.0,2.0),
'technical_41':(-0.6,0.8),
'technical_42':(-1.0,1.0),
'technical_43':(-2.0,0.0),
'technical_6':(-2.0,0.0),
'technical_7':(-0.5,1.5),
'technical_9':(-1.0,0.0),
'y':(-0.10,0.10),
}

info_gain_saved = {'derived_0': -158476443.45271063,
 'fundamental_0': -130569457.88374576,
 'fundamental_18': 21471424.998360336,
 'fundamental_19': 19896893.010529123,
 'fundamental_21': -146104379.8832552,
 'fundamental_33': 20834036.960206702,
 'fundamental_36': 22807284.71794313,
 'fundamental_41': 23584445.207442604,
 'fundamental_42': -158862236.73583558,
 'fundamental_45': 22505366.092955064,
 'fundamental_48': -158813547.74224684,
 'fundamental_53': 23477794.012783736,
 'fundamental_59': -115203272.32681094,
 'fundamental_7': 20967276.974126097,
 'technical_0': -129547495.24981564,
 'technical_11': -111687038.44812873,
 'technical_12': -122955804.98033008,
 'technical_13': -83386904.06984189,
 'technical_14': -149455150.27396062,
 'technical_16': -12213551.645949747,
 'technical_17': -99046594.64685568,
 'technical_18': -58599770.6127148,
 'technical_19': -28313542.973400984,
 'technical_2': -101569741.05200803,
 'technical_20': -73543698.3563616,
 'technical_21': 22032840.209601182,
 'technical_22': -166676505.05062684,
 'technical_24': -111294030.4703038,
 'technical_27': -151232445.2025139,
 'technical_29': -146086127.34213626,
 'technical_3': -109415800.43515168,
 'technical_30': -38034546.78338528,
 'technical_32': -120930393.77632453,
 'technical_33': -120520380.71327816,
 'technical_34': -171667932.74627972,
 'technical_35': -145324445.25380456,
 'technical_36': -153055389.77713448,
 'technical_37': -124431280.62803596,
 'technical_38': -122932025.53370292,
 'technical_39': -112672142.69475743,
 'technical_40': -112057964.21063647,
 'technical_41': -115326374.18407185,
 'technical_42': -74363391.22501546,
 'technical_43': -143923632.5569555,
 'technical_6': -129332318.73648973,
 'technical_7': -89114154.52890107,
 'technical_9': -29732085.14440081,
 'y': -112324397.1178647,
 'y1': 3233094.8465405926}


class FeatureQuantization(object):
	def __init__(self, df, feature_endpoints=feature_endpoints, nbins=30):
		print 'Initialize FeatureQuantization ...',
		self.nbins = nbins
		self.feature_endpoints = feature_endpoints
		self.prob_table = {}
		self.joint_prob_table = {}
		self.step_table = {}
		self.df = df
		n = len(df)
		y1_left, y1_right = -0.10, 0.10
		self.feature_endpoints['y1']=(y1_left, y1_right)
		self.step_table['y1'] = y1_step = (y1_right - y1_left) * 1. / self.nbins
		
		for feature in feature_endpoints:
			print 'quantization of feature %s'%feature
			left, right = feature_endpoints[feature]
			step = (right - left) * 1. / self.nbins
			self.step_table[feature] = step
			counts = [0 for i in range(self.nbins+2)] # (-infty,left), (right,infty)
			joint_counts = [[0 for j in range(self.nbins+2)] for i in range(self.nbins+2)] # (i, j) i represents index of current feature, j represents index of y1
			for idx, feature_val in enumerate(df[feature]):
				quantization_idx = self.get_quantization_index(feature, feature_val)
				y1_val = df.ix[idx, 'y1']
				y1_idx = self.get_quantization_index('y1', y1_val)

				counts[quantization_idx] += 1
				joint_counts[quantization_idx][y1_idx] += 1.0/n
			
			self.prob_table[feature] = [c*1.0/n for c in counts]
			self.joint_prob_table[feature] = joint_counts

		print 'done'

	@classmethod
	def from_pickle(pickle_filename):
		with open(pickle_filename, 'rb') as handle:
			fq = pickle.load(handle)
		return fq

	def to_pickle(self,pickle_filename='feature_quantization.pickle'):
		import pickle
		with open(pickle_filename, 'wb') as handle:
			pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)


	def get_quantization_index(self, feature, feature_val):
		left, right = self.feature_endpoints[feature]
		step = self.step_table[feature]
		if feature_val < left:
			return 0
		elif feature_val >= right:
			return -1
		else:
			return int((feature_val-left)*1./step)

	def get_prob(self, feature, val):
		if '_lag' in feature:
			feature = '_'.join(feature.split('_')[:-1]) # technical_6_lag1 -> technical_6 as lagged feature share same probability as original feature
		quantization_idx = self.get_quantization_index(feature, val)
		return self.prob_table[feature][quantization_idx]
		

	def get_prob_joint_y1(self, feature, feature_val, y1_val):
		quantization_idx = self.get_quantization_index(feature, feature_val)
		y1_idx = self.get_quantization_index('y1', y1_val)
		return self.joint_prob_table[feature][quantization_idx][y1_idx]


	def get_prob_cond_y1(self, feature, feature_val, y1_val):
		'''
		p(z|y) = p(zy) / p(y)
		'''
		p_y1 = self.get_prob('y', y1_val) # self.get_prob('y',y1_val) == self.get_prob('y1',y1_val)
		return self.get_prob_joint_y1(feature, feature_val, y1_val) / p_y1
	

	def build_entropy_table(self):
		print 'building entropy table'
		self.entropy_table = {}
		for feature in self.feature_endpoints:
			entropy = 0
			for val in self.df[feature]:
				entropy += -1* self.get_prob(feature, val) * log(max(1e-7,val),2)
			self.entropy_table[feature] = entropy
		return self.entropy_table

	def build_info_gain_table(self,sample_rate = 0.001):
		'''
		We need to compute all IG(X,y1), where X is feature
		IG(X,y1) = H(X)-H(X|y1) = H(y1) - H(y1|X)
		'''
		print 'building information gain table'
		self.info_gain_table = {}
		df_sample = self.df.sample(frac=sample_rate, replace=False)
		df_sample.reset_index(inplace=True, drop=True)
		print 'sample size is ', len(df_sample)

		for feature in self.feature_endpoints:
			H_feature = self.entropy_table[feature]
			H_feature_cond_y1 = 0
			print 'feature ', feature

			for j in range(len(df_sample)):
				cond_entropy = 0
				y1_val = self.df.ix[j, 'y1']
				
				for i in range(len(df_sample)):
					feature_val = self.df.ix[i, feature]
					prob_feature_cond_y1 = self.get_prob_cond_y1(feature, feature_val,y1_val)
					cond_entropy += -1 * prob_feature_cond_y1 * log(max(1e-7,prob_feature_cond_y1),2)
				H_feature_cond_y1 += self.get_prob('y', y1_val) * cond_entropy
			self.info_gain_table[feature] = H_feature - H_feature_cond_y1 / sample_rate
		return self.info_gain_table

	def build_symmetric_uncertainty_table(self):
		print 'build symmetric uncertainty table'
		self.su_table = {}
		H_y1 = self.entropy_table['y'] # H_y1 == H_y
		for feature in self.feature_endpoints:
			self.su_table[feature] = 2 * self.info_gain_table[feature] / (self.entropy_table[feature] + H_y1)

		return self.su_table

	def rank_su_score(self):
		pass



def test():
	pass


if __name__ == '__main__':
	test()