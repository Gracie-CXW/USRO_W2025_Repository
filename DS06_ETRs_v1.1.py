# This is the updated code for the DS06 extra trees regressor
# models for predicting H2 absorption and diffusion

import pandas as pd
import numpy as np 
from sklearn.ensemble import ExtraTreesRegressor 
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
import sys
import matplotlib.pyplot as plt

''' 
args = argv # input rawdata here
fp = args[1]
'''
fp = r"C:\Users\guacw\Downloads\rawdata (1)\rawdata\Hydrogen.csv" #import filpath here

H2df = pd.read_csv(fp)
diffusion_y = H2df['D_H2 (cm2/s)']
absorption_y = H2df['U_H2 (mol/kg)']
features = H2df.drop(columns=['U_H2 (mol/kg)','D_H2 (cm2/s)','MOF'])

def get_ranks(y):
    y=y
    sorted_y = sorted(y)
    ranks = []
    for n in y:
        ranks.append(sorted_y.index(n))  
    return ranks

def get_SRCC(y,y2):
    y_ranks = get_ranks(y)
    y2_ranks = get_ranks(y2)
    ranks_diff = [p-r for p,r in zip(y_ranks,y2_ranks)]
    d2 = sum([n**2 for n in ranks_diff])
    l = len(y)

    return (1 - (6 * d2 / (l * (l**2 - 1))))

def ETR(y,x,criterion, estimators, maxfeatures, minsamplesleaf, minsamplessplit,scaler):
    y = y
    x = x 
    c = criterion
    e = estimators
    mf = maxfeatures
    msl = minsamplesleaf
    mss = minsamplessplit
    s = scaler

    if s != 'None':
        try:
            if s == 'MaxAbsScaler':
                x=MaxAbsScaler().fit(x).transform(x)
                print('Features have been scaled using MaxAbsScaler')
            elif s=='StandardScaler':
                x=StandardScaler().fit(x).transform(x)
                print('Features have been scaled using Standard Scaler.')
        except:
            print(
                'An error has occured in interpreting the scaling model used. \
                please enter only either "MaxAbsScaler", "StandardScaler", \
                or "None" if no you do not wish to scale your features.'
            )
    
    train_x,test_x,train_y,test_y = train_test_split(
        x,
        y,
        train_size=0.80,
        test_size=0.20,
        random_state=42
    )
    model = ExtraTreesRegressor(
        criterion = c,
        n_estimators = e,
        max_features = mf,
        min_samples_leaf = msl,
        min_samples_split = mss,
        bootstrap = False,
        random_state = 42
    )
    model.fit(train_x,train_y)
    y_pred = model.predict(test_x)
    srcc = get_SRCC(y_pred,test_y)
    r2 = r2_score(y_pred,test_y)
    rmse = np.sqrt(mean_squared_error(y_pred,test_y))
    mae = mean_absolute_error(y_pred,test_y)

    return y_pred,test_y,srcc,r2,rmse,mae


y_pred_d,y_test_d,srcc_d,r2_d,rmse_d,mae_d = ETR(
    y=diffusion_y,
    x=features,
    criterion = 'squared_error',
    estimators = 100,
    maxfeatures = 0.6000000000000001,
    minsamplesleaf = 9,
    minsamplessplit = 19,
    scaler='MaxAbsScaler'
)

y_pred_a,y_test_a,srcc_a,r2_a,rmse_a,mae_a = ETR(
    y=absorption_y,
    x=features,
    criterion = 'squared_error',
    estimators = 100,
    maxfeatures = 0.55,
    minsamplesleaf = 2,
    minsamplessplit = 5,
    scaler='None'
)
print(
    'Diffusion Metrics: \n',
    'rmse: ',rmse_d,'\n',
    'mae: ',mae_d,'\n',
    'SRCC: ',srcc_d,'\n',
    'r2 value: ',r2_d,'\n'
)
print(
    'Absorption Metrics: \n',
    'rmse: ',rmse_a,'\n',
    'mae: ',mae_a,'\n',
    'SRCC: ',srcc_a,'\n',
    'r2 value: ',r2_a,'\n'
)

fig, axs = plt.subplots(1,2)

axs[1,0] = plt.scatter(y_pred_d,y_test_d)
axs[1,1] = plt.scatter(y_pred_a,y_test_a)

plt.show()
