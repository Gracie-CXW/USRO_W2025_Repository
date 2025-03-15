# This is a replication of the Random Forest Regressor model used in the paper DS06.
# This model predicts CH4 diffusion, it uses R2, MAE,SRCC, and RMSE as evaluators. 
# All molecular descriptors are used in the training 
# The paper itself did not specify whether or not features were preprocessed,
# I will use Standard Scaler as I always do. 

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error,mean_squared_error
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import math
import sys
from sys import argv

# Import data and preprocessing
''' 
args = argv # enter data fp
fp = args[1]
'''
fp ="/share_scratch/gwang/USRO_25W/DS06rawdata/Methane.csv" # paste fp here

data = pd.read_csv(fp)
targets = data['D_CH4 (cm2/s)']
features = data.drop(columns=['D_CH4 (cm2/s)','U_CH4 (mol/kg)'])
features = StandardScaler().fit(features.iloc[:,1:]).transform(features.iloc[:,1:])

# function to calculate SRCC(Spearman rank correlation coefficient)
def get_srcc(predicted, real):
    predicted_idx = [i for i,_ in enumerate(predicted)]
    real_idx = [i for i, _ in enumerate(real)]
    ranks_diff = [p-r for p,r in zip(predicted_idx,real_idx)]
    d2 = sum([n**2 for n in ranks_diff])
    n = len(real)

    return (1 - (6 * d2 / (n * (n**2 - 1))))

# sorting each target set from lowest value to highest value (required for SRCC)
def sort_low_to_high(y,x):
    y_x_pairs = list(zip(y.tolist(),x.tolist()))
    y_x_pairs.sort()

    y,x=zip(*y_x_pairs)
    y = list(y)
    x = list(x)

    return y,x
# sorting features and targets
targets,features = sort_low_to_high(targets,features)

# splitting 
train_x,test_x,train_y,test_y = train_test_split(features,targets,train_size=0.80,test_size=0.20,random_state=42)

# model training 
def RFR_CH4(num_trees,max_depth,min_samples_split,min_samples_leaf,num_features):
    nt = num_trees
    md = max_depth
    mss = min_samples_split
    msl = min_samples_leaf
    nf = num_features

    model = RandomForestRegressor(
        n_estimators = nt,
        max_depth = md,
        min_samples_split = mss,
        min_samples_leaf = msl,
        bootstrap = False,
        max_features = nf,
        random_state = 42
    )

    y_pred = model.fit(train_x,train_y).predict(test_x)
    r2 = r2_score(y_pred,test_y)
    rmse = np.sqrt(mean_squared_error(y_pred,test_y))
    mae = mean_absolute_error(y_pred,test_y)
    y_pred_sorted = sorted(y_pred)
    srcc = get_srcc(y_pred_sorted,test_y)

    return r2,rmse,mae,srcc,y_pred

r2,rmse,mae,srcc,y_pred = RFR_CH4(
    num_trees = 1000,
    max_depth = 50,
    min_samples_split = 2,
    min_samples_leaf = 1,
    num_features = 'sqrt'
)

print('R2: ',r2,' RMSE: ',rmse,' MAE: ',mae,' SRCC: ',srcc)

plt.scatter(y_pred,test_y)
plt.show()

