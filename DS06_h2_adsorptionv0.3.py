# ML models for the paper DS06
# notes: for the DS06 paper, model accuracies were evaluated using R2, MAE, SRCC, and RMSE.
# all features were used for the training of the models as no features were analyzed to 
# have correlation r > 0.9.  
# Table 2 lists features used vs ML model accuracy for each model. Only the best preformance
# model will be replicated and tested. 

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
H2adsorp_target = H2data['U_H2 (mol/kg)']
H2adsorp_features = H2data.drop(columns = ['U_H2 (mol/kg)','D_H2 (cm2/s)','MOF'])
# In the paper, H2 adsorption model feature was preprocessed using Max Absolute Scaler
scaler_h2 = MAS().fit(H2adsorp_features.iloc[:,1:])
scaled_features = scaler_h2.transform(H2adsorp_features.iloc[:,1:])
H2adsorp_features.iloc[:,1:] = scaled_features
train_features_h2,test_features_h2,train_target_h2,test_target_h2 = train_test_split(
    H2adsorp_features,H2adsorp_target,train_size = 0.8, test_size = 0.2,
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
def H2_adsorp_model(criterion,estimators,maxfeatures,minsamplesleaf,minsamplessplit):
    c = criterion
    e = estimators
    mf = maxfeatures
    msl = minsamplesleaf
    mss = minsamplessplit
    h2_admodel = ExtraTreesRegressor(bootstrap=False,max_features=mf,min_samples_leaf=msl,
    min_samples_split=mss,criterion=c,n_estimators=e,
    random_state=42)

    h2_admodel.fit(train_features_h2,train_target_h2)
    predicted_train = h2_admodel.predict(train_features_h2)
    predicted_targets = h2_admodel.predict(test_features_h2)
    pred_sorted = sorted(predicted_targets)

    # error calculations
    srcc = get_srcc(pred_sorted,test_target_h2)
    mae = mean_absolute_error(test_target_h2,predicted_targets)
    rmse = root_mean_squared_error(test_target_h2,predicted_targets)
    r2 = r2_score(test_target_h2,predicted_targets)

    rmse_train = root_mean_squared_error(train_target_h2,predicted_train)
    print(rmse_train)
    return srcc,mae,rmse,r2,predicted_targets


parameters = {
    'criterion':['squared_error','absolute_error','friedman_mse','poisson'],
    'estimators':np.arange(50,250,10),
    'maxfeatures':['sqrt','log2',*np.arange(0.1,1,0.05)],
    'minsamplesleaf':np.arange(1,10,1),
    'minsamplessplit':np.arange(2,50,1)
}

srccs=[]
rmses=[]
maes=[]
r2s=[]
params = []
for c in parameters['criterion']:
    for e in parameters['estimators']:
        for mf in parameters['maxfeatures']:
            for msl in parameters['minsamplesleaf']:
                for mss in parameters['minsamplessplit']:
                    srcc,mae,rmse,r2,y = H2_adsorp_model(
                        criterion = c,
                        estimators = e,
                        maxfeatures = mf,
                        minsamplesleaf = msl,
                        minsamplessplit = mss
                    )
                    srccs.append(srcc)
                    rmses.append(rmse)
                    maes.append(mae)
                    r2s.append(r2)
                    params.append([c,e,mf,msl,mss])

idx_min = rmses.index(min(rmses))
print('Minimum RMSE =  ',min(rmses),
'\n MAE = ',maes[idx_min],
'\n SRCC = ',srccs[idx_min],
'\n r2 score = ',r2s[idx_min],
'\n Best Criterion =  ',params[idx_min][0],
'\n Best n_estimators = ',params[idx_min][1],
'\n Best max_features = ',params[idx_min][2],
'\n Best min_samples_leaf = ',params[idx_min][3],
'\n Best min_samples_split = ',params[idx_min][-1])

'''
criterion_ad = ['squared_error','absolute_error','friedman_mse','poisson']
estimators = np.arange(150,400,10,dtype='int32')

srccs=[]
rmses=[]
maes=[]
r2s=[]
params = {}
for c in criterion_ad:
    for x in estimators:
        srcc,mae,rmse,r2,y = H2_adsorp_model(criterion_ad = c, estimators = x)
        srccs.append(srcc)
        rmses.append(rmse)
        maes.append(mae)
        r2s.append(r2)
        params.update({c:x})

idx_min = rmses.index(min(rmses))
#dictionaries r cancer to deal with.
for i, (criterion_dict,est_dict) in enumerate(params.items()):
    if i == idx_min:
        break

print('min rmse ', min(rmses),'\n',
'best criterion: ',criterion_dict,',best n_estimators: ',est_dict,'\n',
'r2 score: ',r2s[idx_min],'\n',
'SRCC: ',srccs[idx_min],'\n',
'MAE = ',maes[idx_min])
'''
