# Same code as DS06_H2_adsoroption, but with H2 Diffusitivty as the target

import sys
from sys import argv
import pandas as pd 
import numpy as np
args = argv       # enter in order, He, H2, N2, CH4 raw data filepaths.

#Initialization 
import sklearn 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler as MAS 
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt 

# function to calculate SRCC(Spearman rank correlation coefficient)
def get_srcc(predicted, real):
    predicted_idx = [i for i,_ in enumerate(predicted)]
    real_idx = [i for i, _ in enumerate(real)]
    ranks_diff = [p-r for p,r in zip(predicted_idx,real_idx)]
    d2 = sum([n**2 for n in ranks_diff])
    n = len(real)

    return (1 - (6 * d2 / (n * (n**2 - 1))))

# H2 Adsorption - Data splitting and preprocessing
from sklearn.ensemble import ExtraTreesRegressor
H2data = pd.read_csv(args[2])
H2diffuse_target = H2data['D_H2 (cm2/s)']
H2_features = H2data.drop(columns = ['U_H2 (mol/kg)','D_H2 (cm2/s)','MOF'])
# In the paper, H2 adsorption model feature was preprocessed using Max Absolute Scaler, 
# need to check if is the case with diffusivity
scaler_h2 = MAS().fit(H2_features.iloc[:,1:])
scaled_features = scaler_h2.transform(H2_features.iloc[:,1:])
H2_features.iloc[:,1:] = scaled_features
train_features_h2,test_features_h2,train_target_h2,test_target_h2 = train_test_split(
    H2_features,H2diffuse_target,train_size = 0.8, test_size = 0.2,
    random_state = 42
) # same random state as in paper

# sorting each target set from lowest value to highest value (required for SRCC)
def sort_low_to_high(y,x):
    y_x_pairs = list(zip(y.values.tolist(),x.values.tolist()))
    y_x_pairs.sort()

    y,x=zip(*y_x_pairs)
    y = list(y)
    x = list(x)

    return y,x

train_target_h2,train_features_h2 = sort_low_to_high(train_target_h2,train_features_h2)
test_target_h2,test_target_h2 = sort_low_to_high(test_target_h2,test_target_h2)

# Model training and testing
def H2_diffuse_model(criterion,estimators):
    criterion = criterion
    estimators = estimators
    h2_model = ExtraTreesRegressor(max_features=0.55,min_samples_leaf=2,
    min_samples_split=5,criterion=criterion,n_estimators=estimators)

    h2_model.fit(train_features_h2,train_target_h2)
    predicted_targets = h2_model.predict(test_features_h2)
    pred_sorted = sorted(predicted_targets)

    # error calculations
    srcc = get_srcc(pred_sorted,test_target_h2)
    mae = mean_absolute_error(predicted_targets,test_target_h2)
    rmse = root_mean_squared_error(predicted_targets,test_target_h2)
    r2 = r2_score(predicted_targets,test_target_h2)

    return srcc,mae,rmse,r2

criterion = ['squared_error','absolute_error','friedman_mse','poisson']
estimators = np.arange(50,250,10,dtype='int32')

srccs=[]
rmses=[]
maes=[]
r2s=[]
params = {}
for c in criterion:
    for x in estimators:
        srcc,mae,rmse,r2 = H2_diffuse_model(criterion = c, estimators = x)
        srccs.append(srcc)
        rmses.append(rmse)
        maes.append(mae)
        r2s.append(r2)
        params.update({c:x})

idx_min = rmses.index(min(rmses))
# dictionaries are cancer to deal with.
for i, (criterion_dict,est_dict) in enumerate(params.items()):
    if i == idx_min:
        break

print('min rmse ', min(rmses),'\n',
'best criterion: ',criterion_dict,',best n_estimators: ',est_dict,'\n',
'r2 score: ',r2s[idx_min],'\n',
'SRCC: ',srccs[idx_min],'\n',
'MAE = ',maes[idx_min])