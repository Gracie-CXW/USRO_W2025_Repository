# This is an attempt at the recreation of the 14 models for the paper DS 22

import pandas as pd
import numpy as np 
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, median_absolute_error
from scipy.stats import kendalltau 
import matplotlib.pyplot as plt 
import sys
from sys import argv
from sklearn.model_selection import GridSearchCV

'''
args = argv     # Put filepath of raw data here: PS, TPS
args[1] = ps_fp
args[2] = tps_fp
'''

ps_fp = r"C:\Users\Grace\Documents\Code\USRO 2025W\Raw_Data\DS22\ps_usable_hydrogen_storage_capacity_gcmcv2.xlsx"
tps_fp = r"C:\Users\Grace\Documents\Code\USRO 2025W\Raw_Data\DS22\tps_usable_hydrogen_storage_capacity_gcmcv2.xlsx"

ps_df = pd.read_excel(ps_fp)
tps_df = pd.read_excel(tps_fp)
ugps = ps_df['UG at PS ']
uvps = ps_df['UV at PS ']
ugtps = tps_df['UG at TPS ']
uvtps = tps_df['UV at TPS ']
# feature combinations chosen based on the Optimal Feature Combinations for each model
# as reported in the paper's SI (Table 5)
ugps_x = ps_df[['GSA ','VF ','PV ','LCD ','PLD ']]
uvps_x = ps_df[['VSA ','VF ','PV ','LCD ','PLD ']]
ugtps_x = tps_df[['Density ','VF ','PV ','LCD ','PLD ']]
uvtps_x = tps_df[['VSA ','VF ','PV ','LCD ','PLD ']]

# 10-fold CV Grid Search for the best hyperparameters for each model; as done in paper
def ETR_search(X,y):
    x = X 
    y = y

    model = ExtraTreesRegressor(random_state=42)

    parameters = {
        'n_estimators':np.arange(50,100,50),
        'criterion':['squared_error','absolute error','friedman_mse','poisson'],
        'max_depth':np.arange(50,150,50),
        'min_samples_split':np.arange(2,20,2),
        'min_samples_leaf':np.arange(1,10,1),
        'max_features':['sqrt','log2',None]
    }

    search = GridSearchCV(model,parameters,cv=10)
    search.fit(x,y)
    best_model = search.best_estimator_

    return best_model.get_params()

def GBR_search(X,y):
    from sklearn.ensemble import GradientBoostingRegressor
    x = X 
    y = y 
    model = GradientBoostingRegressor(random_state=42)

    parameters = {
        'loss':['squared_error','absolute_error','huber','quantile'],
        'learning_rate':np.arange(0.1,0.5,0.2),
        'n_estimators':np.arange(100,200,50),
        'criterion':['friedman_mse','squared_error'],
        'min_sample_split':np.arange(10,200,20),
        'min_samples_leaf':np.arange(10,200,20),
        'max_depth':np.arange(2,10,2),
        'max_features':['sqrt','log2']
    }

    search = GridSearchCV(model,parameters,cv=10)
    search.fit(x,y)
    best_model = search.best_estimator_

    return best_model.get_params()

def Bagging_search(X,y):
    from sklearn.ensemble import BaggingRegressor
    from sklearn.tree import DecisionTreeRegressor
    x = X 
    y = y
    model = BaggingRegressor(random_state=42)

    parametes = {
        'n_estimators':np.arange(10,110,20),
        'max_samples':np.arange(0.05,1.0,0.05)
    }

    search = GridSearchCV(model,parameters,cv=10)
    search.fit(x,y)
    best_model = search.best_estimator_

    return best_model.get_params()

def RF_search(X,y):
    from sklearn.ensemble import RandomForestRegressor
    x = X 
    y = y
    model = RandomForestRegressor(random_state=42)

    parameters = {
        'n_estimators':np.arange(100,5100,500),
        'criterion':['squared_error','absolute_error','friedman_mse','poisson'],
        'max_depth':np.arange(50,500,50)
    }

    search = GridSearchCV(model,parameters,cv=10)
    search.fit(x,y)
    best_model = search.best_estimator_

    return best_model.get_params()

def Bagging_RF_search(X,y):
    from sklearn.ensemble import BaggingRegressor
    from sklearn.ensemble import RandomForestRegressor
    x = X 
    y = y
    model = BaggingRegressor(estimator=RandomForestRegressor(),random_state=42)

    parametes = {
        'n_estimators':np.arange(10,110,20),
        'max_samples':np.arange(0.05,1.0,0.05)
    }

    search = GridSearchCV(model,parameters,cv=10)
    search.fit(x,y)
    best_model = search.best_estimator_

    return best_model.get_params()

def DT_search(X,y):
    from sklearn.tree import DecisionTreeRegressor
    x = X 
    y = y
    model = DecisionTreeRegressor(random_state=42)

    parameters = {
        'criterion':['squared_error','friedman_mse','absolute_error','poisson'],
        'splitter':['best','random'],
        'max_depth':np.arange(10,2000,100),
        'max_features':['sqrt','log2']
    }

    search = GridSearchCV(model,parameters,cv=10)
    search.fit(x,y)
    best_model = search.best_estimator_

    return best_model.get_params()

def NuSVM_search(X,y):
    from sklearn.svm import NuSVR 
    x = X
    y = y
    model = NuSVR(kernel = 'rbf',random_state=42)

    parameters = {
        'nu':np.arange(0.1,1.0,0.1),
        'C':np.arange(0.5,2,0.5),
        'gamma':['scale','auto']
    }

    search = GridSearchCV(model,parameters,cv=10)
    search.fit(x,y)
    best_model = search.best_estimator_

    return best_model.get_params()

def SVR_rbf_search(X,y):
    from sklearn.svm import SVR
    x = X 
    y = y
    model = SVR(kernel='rbf',random_state=42)

    parameters = {
        'gamma':['scale','auto'],
        'C':np.arange(0.5,2.0,0.5),
        'epsilon':np.arange(0.1,1.0,0.1)
    }

    search = GridSearchCV(model,parameters,cv=10)
    search.fit(x,y)
    best_model = search.best_estimator_

    return best_model.get_params()

def SVR_linear_search(X,y):
    from sklearn.svm import SVR
    x = X 
    y = y
    model = SVR(kernel='linear',random_state=42)

    parameters = {
        'C':np.arange(0.5,2.0,0.5),
        'epsilon':np.arange(0.1,1.0,0.1)
    }

    search = GridSearchCV(model,parameters,cv=10)
    search.fit(x,y)
    best_model = search.best_estimator_

    return best_model.get_params()

def LR_search(X,y):
    from sklearn.linear_model import LinearRegression
    x = X 
    y = y
    model = LinearRegression(random_state = 42)

    parameters = {'fit_intercept':[True,False]}

    search = GridSearchCV(model,parameters,cv=10)
    search.fit(x,y)
    best_model = search.best_estimator_

    return best_model.get_params()

def Ridge_search(X,y):
    from sklearn.linear_model import Ridge 
    x = X 
    y = y
    model = Ridge(random_state = 42)
    
    parameters = {
        'alpha':np.arange(0.2,2.0,0.2),
        'fit_intercept':[True,False],
        'solver':['auto','svd','cholesky','lsqr','sparse_cg','sag','saga','lbfgs']
    }

    search = GridSearchCV(model,parameters,cv=10)
    search.fit(x,y)
    best_model = search.best_estimator_

    return best_model.get_params()

def KNN_search(X,y):
    from sklearn.neighbors import KNeighborsRegressor
    x = X 
    y = y
    model = KNeighborsRegressor(randomstate = 42)

    parameters = {
        'n_neighbors':np.arange(10,2100,100),
        'weights':['uniform','distance'],
        'p':[1,2],
        'leaf_size':np.arange(30,1000,130)
    }

    search = GridSearchCV(model,parameters,cv=10)
    search.fit(x,y)
    best_model = search.best_estimator_

    return best_model.get_params()

def AdaBoost_search(X,y):
    from sklearn.ensemble import AdaBoostRegressor
    x = X 
    y = y
    model = AdaBoostRegressor(random_state = 42)

    parameters = {
        'n_estimators':np.arange(50,500,50),
        'learning_rate':np.arange(0.5,2.5,0.5),
        'loss':['linear','square','exponential']
    }

    search = GridSearchCV(model,parameters,cv=10)
    search.fit(x,y)
    best_model = search.best_estimator_

    return best_model.get_params()


