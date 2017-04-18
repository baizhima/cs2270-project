#from data_reader import *
from train import main

df = main()
corr_matrix = df.corr(method='pearson')
