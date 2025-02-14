# This code is for the purpose of replicating the Random Forest Regressor ML model from the paper DS15
# metrics to evaluate ML model preformance included r2, r, RMSE, and MAPE. 
# In the paper, it was specified that training the model using chemical features in groups A+C yield the best results
# i.e. lowest RMSE and highest R2. 
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
from sys import argv
import sys 
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

# Import data 
args = argv     # Enter here the filepath to the raw data.csv, or 
fp = args[1]
'''
fp = # input filepath
'''
Data = pd.read_csv(fp,delimiter = ',')
targets = Data['S']
features = Data.drop(columns = ['S','OMS Type'])

# Data preprocessing, I am using standard scaler although the paper did not specify preprocessing of the features. 
from sklearn import preprocessing

scaler = preprocessing.StandardScaler()
scaled_features = scaler.fit(features.iloc[:,1:]).transform(features.iloc[:,1:])
targets = targets.to_list()
scaled_targets = []

for n in targets:
    scaled_targets.append(math.log10(n))

from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y = train_test_split(
    scaled_features,
    scaled_targets,
    train_size = 0.80,
    test_size=0.20,
    random_state=42)

# Model building
def RFR(max_depth,num_trees,min_per_node,min_samples_leaf,max_features):
    m = max_depth
    nt = num_trees
    mpn = min_per_node
    msl = min_samples_leaf
    mf = max_features

    model = RandomForestRegressor(
        n_estimators= nt,
        max_depth=m,
        min_samples_leaf= msl,
        min_samples_split=mpn,
        bootstrap=False,
        max_features=mf)
    model.fit(train_x,train_y)

    pred_y = model.predict(test_x)
    
    r2 = r2_score(test_y,pred_y)
    rmse = np.sqrt(mean_squared_error(test_y,pred_y))
    mape = mean_absolute_percentage_error(test_y,pred_y)

    return r2,rmse,mape,pred_y

# Start by testing paper reported parameters
r2,rmse,mape,y_pred = RFR(max_depth=50,num_trees=1000,min_per_node=2,min_samples_leaf=1,max_features='sqrt')

print("R2 score: ",r2," RMSE: ",rmse," MAPE: ",mape)