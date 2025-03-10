# This is an attempt at the recreation of the 14 models for the paper DS 22
# For the UV/PS output
# Evaluation metrics are: R2, AUE, RMSE, Kendal r
import pandas as pd 
import numpy as np 
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from scipy.stats import kendalltau
import matplotlib.pyplot as plt 

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

def ETR(X,y):
    from sklearn.ensemble import ExtraTreesRegressor
    x = X 
    y = y
    train_x, test_x, train_y, test_y = train_test_split(x,y,train_size = 0.80, test_size = 0.20, random_state = 42)
    model = ExtraTreesRegressor(
        random_state = 42,
        criterion = 'poisson',
        max_depth = 50,
        max_features = 'sqrt',
        min_samples_leaf = 2,
        min_samples_split = 12,
        n_estimators = 50
    ) # Gridsearch Parameters here
    model.fit(train_x,train_y)

    pred_y = model.predict(test_x)
    r2 = r2_score(test_y,pred_y)
    mae = mean_absolute_error(test_y,pred_y)
    rmse = np.sqrt(mean_squared_error(test_y,pred_y))
    kr = kendalltau(test_y,pred_y)

    return pred_y,r2,mae,rmse,kr

def GBR(X,y):
    from sklearn.ensemble import GradientBoostingRegressor
    x = X 
    y = y 
    train_x, test_x, train_y, test_y = train_test_split(x,y,train_size = 0.80, test_size = 0.20, random_state = 42)
    model = GradientBoostingRegressor(
        random_state = 42
    ) # Gridsearch Parameters here

    model.fit(train_x,train_y)

    pred_y = model.predict(test_x)
    r2 = r2_score(test_y,pred_y)
    mae = mean_absolute_error(test_y,pred_y)
    rmse = np.sqrt(mean_squared_error(test_y,pred_y))
    kr = kendalltau(test_y,pred_y)

    return pred_y,r2,mae,rmse,kr

def Bagging(X,y):
    from sklearn.ensemble import BaggingRegressor
    x = X 
    y = y 
    train_x, test_x, train_y, test_y = train_test_split(x,y,train_size = 0.80, test_size = 0.20, random_state = 42)
    model = BaggingRegressor(
        random_state = 42,
        n_estimators = 90,
        max_samples = 0.25
    ) # Gridsearch Parameters here

    model.fit(train_x,train_y)

    pred_y = model.predict(test_x)
    r2 = r2_score(test_y,pred_y)
    mae = mean_absolute_error(test_y,pred_y)
    rmse = np.sqrt(mean_squared_error(test_y,pred_y))
    kr = kendalltau(test_y,pred_y)

    return pred_y,r2,mae,rmse,kr

def RF(X,y):
    from sklearn.ensemble import RandomForestRegressor
    x = X 
    y = y
    train_x, test_x, train_y, test_y = train_test_split(x,y,train_size = 0.80, test_size = 0.20, random_state = 42)
    model = RandomForestRegressor(
        random_state = 42
    ) # Gridsearch Parameters here 

    model.fit(train_x,train_y)

    pred_y = model.predict(test_x)
    r2 = r2_score(test_y,pred_y)
    mae = mean_absolute_error(test_y,pred_y)
    rmse = np.sqrt(mean_squared_error(test_y,pred_y))
    kr = kendalltau(test_y,pred_y)

    return pred_y,r2,mae,rmse,kr

def Bagging_RF(X,y):
    from sklearn.ensemble import BaggingRegressor
    from sklearn.ensemble import RandomForestRegressor
    x = X 
    y = y
    train_x, test_x, train_y, test_y = train_test_split(x,y,train_size = 0.80, test_size = 0.20, random_state = 42)
    model = BaggingRegressor(
        estimator=RandomForestRegressor(),
        random_state=42
    ) # Gridsearch Parameters here 

    model.fit(train_x,train_y)

    pred_y = model.predict(test_x)
    r2 = r2_score(test_y,pred_y)
    mae = mean_absolute_error(test_y,pred_y)
    rmse = np.sqrt(mean_squared_error(test_y,pred_y))
    kr = kendalltau(test_y,pred_y)

    return pred_y,r2,mae,rmse,kr

def DT(X,y):
    from sklearn.tree import DecisionTreeRegressor
    x = X 
    y = y
    train_x, test_x, train_y, test_y = train_test_split(x,y,train_size = 0.80, test_size = 0.20, random_state = 42)
    model = DecisionTreeRegressor(
        random_state=42,
        criterion = 'absolute_error',
        splitter = 'best',
        max_depth = 10,
        max_features = 'sqrt'
    ) # Gridsearch Parameters here 

    model.fit(train_x,train_y)

    pred_y = model.predict(test_x)
    r2 = r2_score(test_y,pred_y)
    mae = mean_absolute_error(test_y,pred_y)
    rmse = np.sqrt(mean_squared_error(test_y,pred_y))
    kr = kendalltau(test_y,pred_y)

    return pred_y,r2,mae,rmse,kr

def NuSVM(X,y):
    from sklearn.svm import NuSVR 
    x = X
    y = y
    train_x, test_x, train_y, test_y = train_test_split(x,y,train_size = 0.80, test_size = 0.20, random_state = 42)
    model = NuSVR(
        kernel = 'rbf'
    ) # Gridsearch Parameters here 

    model.fit(train_x,train_y)

    pred_y = model.predict(test_x)
    r2 = r2_score(test_y,pred_y)
    mae = mean_absolute_error(test_y,pred_y)
    rmse = np.sqrt(mean_squared_error(test_y,pred_y))
    kr = kendalltau(test_y,pred_y)

    return pred_y,r2,mae,rmse,kr

def SVR_rbf(X,y):
    from sklearn.svm import SVR
    x = X 
    y = y 
    train_x, test_x, train_y, test_y = train_test_split(x,y,train_size = 0.80, test_size = 0.20, random_state = 42)
    model = SVR(
        kernel='rbf'
    ) # Gridsearch Parameters here 

    model.fit(train_x,train_y)

    pred_y = model.predict(test_x)
    r2 = r2_score(test_y,pred_y)
    mae = mean_absolute_error(test_y,pred_y)
    rmse = np.sqrt(mean_squared_error(test_y,pred_y))
    kr = kendalltau(test_y,pred_y)

    return pred_y,r2,mae,rmse,kr

def SVR_linear(X,y):
    from sklearn.svm import svr
    x = X 
    y = y 
    train_x, test_x, train_y, test_y = train_test_split(x,y,train_size = 0.80, test_size = 0.20, random_state = 42)
    model = SVR(
        kernel = 'linear'
    ) # Gridsearch Parameters here

    model.fit(train_x,train_y)

    pred_y = model.predict(test_x)
    r2 = r2_score(test_y,pred_y)
    mae = mean_absolute_error(test_y,pred_y)
    rmse = np.sqrt(mean_squared_error(test_y,pred_y))
    kr = kendalltau(test_y,pred_y)

    return pred_y,r2,mae,rmse,kr

def LR(X,y):
    from sklearn.linear_model import LinearRegression
    x = X 
    y = y 
    train_x, test_x, train_y, test_y = train_test_split(x,y,train_size = 0.80, test_size = 0.20, random_state = 42)
    model = LinearRegression(
        fit_intercept = True
    ) # Gridsearch Parameters here

    model.fit(train_x,train_y)

    pred_y = model.predict(test_x)
    r2 = r2_score(test_y,pred_y)
    mae = mean_absolute_error(test_y,pred_y)
    rmse = np.sqrt(mean_squared_error(test_y,pred_y))
    kr = kendalltau(test_y,pred_y)

    return pred_y,r2,mae,rmse,kr

def Ridge(X,y):
    from sklearn.linear_model import Ridge
    x = X 
    y = y 
    train_x, test_x, train_y, test_y = train_test_split(x,y,train_size = 0.80, test_size = 0.20, random_state = 42)
    model = Ridge(
        random_state = 42,
        alpha = 1.8,
        fit_intercept = True,
        solver = 'svd'
    ) # Gridsearch Parameters here

    model.fit(train_x,train_y)

    pred_y = model.predict(test_x)
    r2 = r2_score(test_y,pred_y)
    mae = mean_absolute_error(test_y,pred_y)
    rmse = np.sqrt(mean_squared_error(test_y,pred_y))
    kr = kendalltau(test_y,pred_y)

    return pred_y,r2,mae,rmse,kr

def KNN(X,y):
    from sklearn.neighbors import KNeighborsRegressor
    x = X 
    y = y 
    train_x, test_x, train_y, test_y = train_test_split(x,y,train_size = 0.80, test_size = 0.20, random_state = 42)
    model = KNeighborsRegressor(
        n_neighbors = 10,
        weights = 'distance',
        p = 1,
        leaf_size = 810
    ) # Gridsearch Parameters here

    model.fit(train_x,train_y)

    pred_y = model.predict(test_x)
    r2 = r2_score(test_y,pred_y)
    mae = mean_absolute_error(test_y,pred_y)
    rmse = np.sqrt(mean_squared_error(test_y,pred_y))
    kr = kendalltau(test_y,pred_y)

    return pred_y,r2,mae,rmse,kr

def AdaBoost(X,y):
    from sklearn.ensemble import AdaBoostRegressor
    x = X 
    y = y 
    train_x, test_x, train_y, test_y = train_test_split(x,y,train_size = 0.80, test_size = 0.20, random_state = 42)
    model = AdaBoostRegressor(
        random_state = 42,
        n_estimators = 50,
        learning_rate = 0.5,
        loss = 'exponential'
    ) # Gridsearch Parameters here

    model.fit(train_x,train_y)

    pred_y = model.predict(test_x)
    r2 = r2_score(test_y,pred_y)
    mae = mean_absolute_error(test_y,pred_y)
    rmse = np.sqrt(mean_squared_error(test_y,pred_y))
    kr = kendalltau(test_y,pred_y)

    return pred_y,r2,mae,rmse,kr

# Getting Metrics from models - first batch
models = [
    'ETR','Bagging','DT',
    'KNN','LR','Ridge',
    'AdaBoost'
]

for model_name in models:
    if model_name in globals():
        model = globals()[model_name]
        pred_y,r2,mae,rmse,kr = model(uvps_x,uvps)
        print(
            f'\n The evaluation metrics for the following model: {model_name} are:\n',
            f'r2: ',r2,'\n',
            f'mae: ',mae,'\n',
            f'rmse: ',rmse,'\n',
            f'kendalltau: ',kr,'\n'
        )
    else:
        print(f"Error: Function '{model_name}' is not defined.")



