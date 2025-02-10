#XG Boost Model for the Paper DS15
# This paper used different combinations of features for model training. Only the combination
# reported with the best r^2 and lest error is used here.
# The paper uses RMSE, MAPE, r and r^2 are evaluating factors. 
# The ideal features for training exclude: MA,DM,OMS index,epsilon difference,sigma
# Target is CO2-to-CO selectivity, indicated as 'S' in raw data.
import sys
from sys import argv
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error, root_mean_squared_error, r2_score
import matplotlib.pyplot as plt
import math

#importing raw data, and splitting.
args = argv      # input data filepath here, should be .csv format

Dataset = pd.read_csv(args[1])
targets = Dataset['S']
features = Dataset.drop(columns = ['S','OMS type'])

# Data preprocessing - paper did not specify whether the features were preprocessed
# prior to training, nor the type of preprocessing for each feature. 
# I am using standard scaler

from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
scaled_features = scaler.fit(features.iloc[:, 1:]).transform(features.iloc[:, 1:])
targets = targets.to_list()
scaled_targets = []

for n in targets:
    scaled_targets.append(math.log10(n))

from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y = train_test_split(
    scaled_features,
    scaled_targets,
    train_size = 0.80, 
    test_size = 0.20, 
    random_state = 42)

# Model building
import xgboost
from xgboost import XGBRegressor

xgb_model = XGBRegressor(
    learning_rate = 0.05, 
    max_depth = 10, 
    n_estimators = 1500,
    random_state = 42)

xgb_model.fit(train_x,train_y)

pred_y = xgb_model.predict(test_x)
print('RMSE = ',root_mean_squared_error(test_y,pred_y),
'MAPE = ',mean_absolute_percentage_error(test_y,pred_y),
'r2 score = ',r2_score(test_y,pred_y))

plt.scatter(pred_y,test_y)
plt.show()