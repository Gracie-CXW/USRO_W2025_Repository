# This is the updated code for the DS06 extra trees regressor
# models for predicting H2 absorption and diffusion

import pandas as pd
import numpy as np 
from sklearn.ensemble import ExtraTreesRegressor 
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
import matplotlib.pyplot as plt

# Importing raw data via Command Line Arguments
''' 
args = argv # input rawdata here
fp = args[1]
'''
#importing raw data via Hardcoded Paths
fp = r"C:\Users\Grace\Documents\Code\USRO 2025W\Raw_Data\DS06\rawdata\Hydrogen.csv" #import filpath here
ASR_bad = r"C:\Users\Grace\Documents\Code\USRO 2025W\Good_Bad_MOF_splits\DS06\Bad\DS06_ASR_bad_output_H2.txt"
FSR_bad = r"C:\Users\Grace\Documents\Code\USRO 2025W\Good_Bad_MOF_splits\DS06\Bad\DS06_FSR_bad_output_H2.txt"
ASR_good = r"C:\Users\Grace\Documents\Code\USRO 2025W\Good_Bad_MOF_splits\DS06\Good\DS06_ASR_good_output_H2.txt"
FSR_good = r"C:\Users\Grace\Documents\Code\USRO 2025W\Good_Bad_MOF_splits\DS06\Good\DS06_FSR_good_output_H2.txt"

# Data splitting 
# For training with Bad
output = r"C:\Users\Grace\Documents\Code\USRO 2025W\My Code\DS06_data\H2_adsorption\Train_bad.txt"

asr_bad = open(ASR_bad,'r').read().splitlines()
fsr_bad = open(FSR_bad,'r').read().splitlines()
bad_names = set(asr_bad+fsr_bad)
asr_good = open(ASR_good,'r').read().splitlines()
fsr_good = open(FSR_good,'r').read().splitlines()
good_names = set(asr_good+fsr_good)

H2df = pd.read_csv(fp)
H2_bad = H2df[H2df['MOF'].isin(bad_names)]

H2_bad_x = H2_bad.drop(columns = ['U_H2 (mol/kg)','D_H2 (cm2/s)'])
H2_bad_y_D = H2_bad['D_H2 (cm2/s)']
H2_bad_y_A = H2_bad['U_H2 (mol/kg)']

H2_good = H2df[H2df['MOF'].isin(good_names)]
H2_good_x = H2_good.drop(columns = ['U_H2 (mol/kg)','D_H2 (cm2/s)'])
H2_good_y_D = H2_good['D_H2 (cm2/s)']
H2_good_y_A = H2_good['U_H2 (mol/kg)']

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

def ETR_dif(train_x,test_x,train_y,test_y):

    train_x = train_x 
    train_y = train_y 
    test_x = test_x 
    test_y = test_y

    scaler = StandardScaler().fit(train_x)
    train_x = scaler.transform(train_x)
    test_x = scaler.transform(test_x)

    model = ExtraTreesRegressor(
        criterion = 'squared_error',
        n_estimators = 100,
        max_features = 0.6000000000000001,
        min_samples_leaf = 9,
        min_samples_split = 19,
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

def ETR_ads(train_x,test_x,train_y,test_y):
    
    train_x = train_x 
    train_y = train_y 
    test_x = test_x 
    test_y = test_y 

    model = ExtraTreesRegressor(
        criterion = 'squared_error',
        n_estimators = 100,
        max_features = 0.55,
        min_samples_leaf = 2,
        min_samples_split = 5,
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

# For now, train = bad, test = bad,good,mixed.
# For this paper, the mixed dataset is just the original dataset, since all data was sourced from the CoRE 2019 Dataset
# This is for the diffusion output
tests = ['H2_bad','H2_good','H2_mixed']
def diffusion_bad(datasets):
    datasets = datasets
    for dataset in datasets:
        if dataset == 'H2_bad':
            # Data Splitting and Training
            train_x,test_x,train_y,test_y = train_test_split(H2_bad_x,H2_bad_y_D,train_size = 0.80,test_size = 0.20)

            y_pred_D,test_y_D,srcc_D,r2_D,rmse_D,mae_D = ETR_dif(
                train_x.drop(columns=['MOF']),
                test_x.drop(columns=['MOF']),
                train_y,test_y
                )
            
            #Reporting metrics
            
            with open(output,'a') as results:
                results.write('\n Metrics for H2 Diffusion. (bad/bad) \n')
                results.write('\n Number of MOFs used for training (bad): ' + str(len(train_x)))
                results.write('\n List of MOFs used for training (bad): \n' + train_x['MOF'].to_string())
                results.write('\n Number of MOFs used for testing (bad): ' + str(len(test_x)))
                results.write('\n List of MOFs used for testing (bad): \n' + test_x['MOF'].to_string())
                results.write('\n Metrics: (Train = Bad, Test = Bad, Split = 80/20) \n')
                results.write(f'SRCC: {srcc_D}\n')
                results.write(f'R² Score: {r2_D}\n')
                results.write(f'RMSE: {rmse_D}\n')
                results.write(f'MAE: {mae_D}\n')
                results.write('-' * 50 + '\n')
            
            # plotting the correlation graph
            fig = plt.figure(figsize=(7,7),dpi=100)
            plt.rc('axes',axisbelow=True)
            plt.rcParams['font.size'] = 12
            plt.rcParams['font.family'] = "sans-serif"
            plt.grid(color='lightgrey',linestyle='-')
            plt.ylim(ymin=0,ymax=0.0070)
            plt.xlim(xmin=0,xmax=0.0070)
            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8)

            y = np.arange(0,0.01,0.0070)
            x = np.arange(0,0.01,0.0070)
            plt.plot(x,y,color='darkgoldenrod',linestyle='--',alpha=0.5)
            plt.scatter(y_pred_D,test_y_D,color='chocolate')

            plt.xlabel('Predicted H2 Diffusivity (cm2/s)')
            plt.ylabel('Real H2 Diffusivity (cm2/s)')
            plt.title('Predicted vs Real H2 Diffusivities. Train=Bad, Test=Bad',y=1.03)
            text = f'SRCC: {srcc_D:.4f} \n R2: {r2_D:.4f} \n RMSE: {rmse_D:.5f} \n MAE: {mae_D:.5f}'
            plt.text(x=0.004,y=0.0005,s=text)
            plt.savefig(r"C:\Users\Grace\Documents\Code\USRO 2025W\My Code\DS06_data\H2_diffusion\Correlation_Plots\H2_bad_bad.png")
            plt.close()


        elif dataset == 'H2_good':
            y_pred_D,test_y_D,srcc_D,r2_D,rmse_D,mae_D = ETR_dif(
                H2_bad_x.drop(columns=['MOF']),
                H2_good_x.drop(columns=['MOF']),
                H2_bad_y_D,H2_good_y_D
                )

            with open(output,'a') as results:
                results.write('\n Metrics for H2 Diffusion. (bad/good) \n')
                results.write('\n Number of MOFs used for training(bad): ' + str(len(H2_bad_x)))
                results.write('\n List of MOFs used for training (bad): \n' + H2_bad_x['MOF'].to_string())
                results.write('\n Number of MOFs used for testing (good): '+ str(len(H2_good_y_D)))
                results.write('\n List of MOFs used for testing (good): \n' + H2_good_x['MOF'].to_string())
                results.write('\n Metrics: (Train = Bad, Test = Good, No split.) \n')
                results.write(f'SRCC: {srcc_D}\n')
                results.write(f'R² Score: {r2_D}\n')
                results.write(f'RMSE: {rmse_D}\n')
                results.write(f'MAE: {mae_D}\n')
                results.write('-' * 50 + '\n')
            
            # plotting the correlation graph
            fig = plt.figure(figsize=(7,7),dpi=100)
            plt.rc('axes',axisbelow=True)
            plt.rcParams['font.size'] = 12
            plt.rcParams['font.family'] = "sans-serif"
            plt.grid(color='lightgrey',linestyle='-')
            plt.ylim(ymin=0,ymax=0.0070)
            plt.xlim(xmin=0,xmax=0.0070)
            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8)

            y = np.arange(0,0.01,0.0070)
            x = np.arange(0,0.01,0.0070)
            plt.plot(x,y,color='indianred',linestyle='--',alpha=0.5)
            plt.scatter(y_pred_D,test_y_D,color='firebrick')

            plt.xlabel('Predicted H2 Diffusivity (cm2/s)')
            plt.ylabel('Real H2 Diffusivity (cm2/s)')
            plt.title('Predicted vs Real H2 Diffusivities. Train=Bad, Test=Good',y=1.03)
            text = f'SRCC: {srcc_D:.4f} \n R2: {r2_D:.4f} \n RMSE: {rmse_D:.5f} \n MAE: {mae_D:.5f}'
            plt.text(x=0.004,y=0.0005,s=text)
            plt.savefig(r"C:\Users\Grace\Documents\Code\USRO 2025W\My Code\DS06_data\H2_diffusion\Correlation_Plots\H2_bad_good.png")
            plt.close()


        elif dataset == 'H2_mixed':
            train_x,test_bad_x,train_y,test_bad_y_D = train_test_split(H2_bad_x,H2_bad_y_D,train_size=0.80,test_size=0.20)

            test_good_x = H2_good_x.sample(n=len(test_bad_x))
            test_good_y_D = H2_good_y_D.sample(n=len(test_bad_y_D))
            test_x = pd.concat([test_bad_x,test_good_x])
            test_y_D = pd.concat([test_bad_y_D,test_good_y_D])

            y_pred_D,test_y_D,srcc_D,r2_D,rmse_D,mae_D = ETR_dif(
                train_x.drop(columns=['MOF']),
                test_x.drop(columns=['MOF']),
                train_y,test_y_D
                )

            with open(output,'a') as results:
                results.write('\n Metrics for H2 Diffusion. (bad/mixed) \n')
                results.write('\n Number of MOFs used for training (bad): ' + str(len(train_x)))
                results.write('\n List of MOFs used for training (bad): \n' + train_x['MOF'].to_string())
                results.write('\n Number of MOFs used for testing (mixed): ' + str(len(test_x)))
                results.write('\n List of MOFs used for testing (mixed): \n' + test_x['MOF'].to_string())
                results.write('\n Metrics: (Train = Bad, Test = Mixed, Split = 80 (train), 20(n) bad + n good (test)) \n')
                results.write(f'SRCC: {srcc_D}\n')
                results.write(f'R² Score: {r2_D}\n')
                results.write(f'RMSE: {rmse_D}\n')
                results.write(f'MAE: {mae_D}\n')
                results.write('-' * 50 + '\n')
            
            # plotting the correlation graph
            fig = plt.figure(figsize=(7,7),dpi=100)
            plt.rc('axes',axisbelow=True)
            plt.rcParams['font.size'] = 12
            plt.rcParams['font.family'] = "sans-serif"
            plt.grid(color='lightgrey',linestyle='-')
            plt.ylim(ymin=0,ymax=0.0070)
            plt.xlim(xmin=0,xmax=0.0070)
            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8)

            y = np.arange(0,0.01,0.0070)
            x = np.arange(0,0.01,0.0070)
            plt.plot(x,y,color='sandybrown',linestyle='--',alpha=0.5)
            plt.scatter(y_pred_D,test_y_D,color='darkorange')

            plt.xlabel('Predicted H2 Diffusivity (cm2/s)')
            plt.ylabel('Real H2 Diffusivity (cm2/s)')
            plt.title('Predicted vs Real H2 Diffusivities. Train=Bad, Test=Mixed',y=1.03)
            text = f'SRCC: {srcc_D:.4f} \n R2: {r2_D:.4f} \n RMSE: {rmse_D:.5f} \n MAE: {mae_D:.5f}'
            plt.text(x=0.004,y=0.0005,s=text)
            plt.savefig(r"C:\Users\Grace\Documents\Code\USRO 2025W\My Code\DS06_data\H2_diffusion\Correlation_Plots\H2_bad_mixed.png")
            plt.close()

def diffusion_good(datasets):
    datasets = datasets
    for dataset in datasets:
        if dataset == 'H2_good':
            # Data Splitting and Training
            train_x,test_x,train_y,test_y = train_test_split(H2_good_x,H2_good_y_D,train_size = 0.80,test_size = 0.20)

            y_pred_D,test_y_D,srcc_D,r2_D,rmse_D,mae_D = ETR_dif(
                train_x.drop(columns=['MOF']),
                test_x.drop(columns=['MOF']),
                train_y,test_y
                )
            
            #Reporting metrics
            
            with open(output,'a') as results:
                results.write('\n Metrics for H2 Diffusion. (good/good) \n')
                results.write('\n Number of MOFs used for training (good): ' + str(len(train_x)))
                results.write('\n List of MOFs used for training (good): \n' + train_x['MOF'].to_string())
                results.write('\n Number of MOFs used for testing (good): ' + str(len(test_x)))
                results.write('\n List of MOFs used for testing (good): \n' + test_x['MOF'].to_string())
                results.write('\n Metrics: (Train = Good, Test = Good, Split = 80/20) \n')
                results.write(f'SRCC: {srcc_D}\n')
                results.write(f'R² Score: {r2_D}\n')
                results.write(f'RMSE: {rmse_D}\n')
                results.write(f'MAE: {mae_D}\n')
                results.write('-' * 50 + '\n')
            
            # plotting the correlation graph
            fig = plt.figure(figsize=(7,7),dpi=100)
            plt.rc('axes',axisbelow=True)
            plt.rcParams['font.size'] = 12
            plt.rcParams['font.family'] = "sans-serif"
            plt.grid(color='lightgrey',linestyle='-')
            plt.ylim(ymin=0,ymax=0.0070)
            plt.xlim(xmin=0,xmax=0.0070)
            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8)

            y = np.arange(0,0.01,0.0070)
            x = np.arange(0,0.01,0.0070)
            plt.plot(x,y,color='cadetblue',linestyle='--',alpha=0.5)
            plt.scatter(y_pred_D,test_y_D,color='teal')

            plt.xlabel('Predicted H2 Diffusivity (cm2/s)')
            plt.ylabel('Real H2 Diffusivity (cm2/s)')
            plt.title('Predicted vs Real H2 Diffusivities. Train=Good, Test=Good',y=1.03)
            text = f'SRCC: {srcc_D:.4f} \n R2: {r2_D:.4f} \n RMSE: {rmse_D:.5f} \n MAE: {mae_D:.5f}'
            plt.text(x=0.004,y=0.0005,s=text)
            plt.savefig(r"C:\Users\Grace\Documents\Code\USRO 2025W\My Code\DS06_data\H2_diffusion\Correlation_Plots\H2_good_good.png")
            plt.close()
        
        elif dataset =='H2_bad':
            y_pred_D,test_y_D,srcc_D,r2_D,rmse_D,mae_D = ETR_dif(
                H2_good_x.drop(columns=['MOF']),
                H2_bad_x.drop(columns=['MOF']),
                H2_good_y_D,H2_bad_y_D
                )

            with open(output,'a') as results:
                results.write('\n Metrics for H2 Diffusion. (good/bad) \n')
                results.write('\n Number of MOFs used for training(good): ' + str(len(H2_good_x)))
                results.write('\n List of MOFs used for training (good): \n' + H2_good_x['MOF'].to_string())
                results.write('\n Number of MOFs used for testing (bad): '+ str(len(H2_bad_x)))
                results.write('\n List of MOFs used for testing (bad): \n' + H2_bad_x['MOF'].to_string())
                results.write('\n Metrics: (Train = Good, Test = Bad, No split.) \n')
                results.write(f'SRCC: {srcc_D}\n')
                results.write(f'R² Score: {r2_D}\n')
                results.write(f'RMSE: {rmse_D}\n')
                results.write(f'MAE: {mae_D}\n')
                results.write('-' * 50 + '\n')
            
            # plotting the correlation graph
            fig = plt.figure(figsize=(7,7),dpi=100)
            plt.rc('axes',axisbelow=True)
            plt.rcParams['font.size'] = 12
            plt.rcParams['font.family'] = "sans-serif"
            plt.grid(color='lightgrey',linestyle='-')
            plt.ylim(ymin=0,ymax=0.0070)
            plt.xlim(xmin=0,xmax=0.0070)
            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8)

            y = np.arange(0,0.01,0.0070)
            x = np.arange(0,0.01,0.0070)
            plt.plot(x,y,color='slateblue',linestyle='--',alpha=0.5)
            plt.scatter(y_pred_D,test_y_D,color='darkslateblue')

            plt.xlabel('Predicted H2 Diffusivity (cm2/s)')
            plt.ylabel('Real H2 Diffusivity (cm2/s)')
            plt.title('Predicted vs Real H2 Diffusivities. Train=Good, Test=Bad',y=1.03)
            text = f'SRCC: {srcc_D:.4f} \n R2: {r2_D:.4f} \n RMSE: {rmse_D:.5f} \n MAE: {mae_D:.5f}'
            plt.text(x=0.004,y=0.0005,s=text)
            plt.savefig(r"C:\Users\Grace\Documents\Code\USRO 2025W\My Code\DS06_data\H2_diffusion\Correlation_Plots\H2_good_bad.png")
            plt.close()
        
        elif dataset =='H2_mixed':
            train_x,test_good_x,train_y,test_good_y_D = train_test_split(H2_good_x,H2_good_y_D,train_size=0.80,test_size=0.20)

            test_bad_x = H2_bad_x.sample(n=len(test_good_x))
            test_bad_y_D = H2_bad_y_D.sample(n=len(test_good_y_D))
            test_x = pd.concat([test_bad_x,test_good_x])
            test_y_D = pd.concat([test_bad_y_D,test_good_y_D])

            y_pred_D,test_y_D,srcc_D,r2_D,rmse_D,mae_D = ETR_dif(
                train_x.drop(columns=['MOF']),
                test_x.drop(columns=['MOF']),
                train_y,test_y_D
                )

            with open(output,'a') as results:
                results.write('\n Metrics for H2 Diffusion. (good/mixed) \n')
                results.write('\n Number of MOFs used for training (good): ' + str(len(train_x)))
                results.write('\n List of MOFs used for training (good): \n' + train_x['MOF'].to_string())
                results.write('\n Number of MOFs used for testing (mixed): ' + str(len(test_x)))
                results.write('\n List of MOFs used for testing (mixed): \n' + test_x['MOF'].to_string())
                results.write('\n Metrics: (Train = Good, Test = Mixed, Split = 80 (train), 20(n) good + n bad (test)) \n')
                results.write(f'SRCC: {srcc_D}\n')
                results.write(f'R² Score: {r2_D}\n')
                results.write(f'RMSE: {rmse_D}\n')
                results.write(f'MAE: {mae_D}\n')
                results.write('-' * 50 + '\n')
            
            # plotting the correlation graph
            fig = plt.figure(figsize=(7,7),dpi=100)
            plt.rc('axes',axisbelow=True)
            plt.rcParams['font.size'] = 12
            plt.rcParams['font.family'] = "sans-serif"
            plt.grid(color='lightgrey',linestyle='-')
            plt.ylim(ymin=0,ymax=0.0070)
            plt.xlim(xmin=0,xmax=0.0070)
            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8)

            y = np.arange(0,0.01,0.0070)
            x = np.arange(0,0.01,0.0070)
            plt.plot(x,y,color='plum',linestyle='--',alpha=0.5)
            plt.scatter(y_pred_D,test_y_D,color='mediumvioletred')

            plt.xlabel('Predicted H2 Diffusivity (cm2/s)')
            plt.ylabel('Real H2 Diffusivity (cm2/s)')
            plt.title('Predicted vs Real H2 Diffusivities. Train=Good, Test=Mixed',y=1.03)
            text = f'SRCC: {srcc_D:.4f} \n R2: {r2_D:.4f} \n RMSE: {rmse_D:.5f} \n MAE: {mae_D:.5f}'
            plt.text(x=0.004,y=0.0005,s=text)
            plt.savefig(r"C:\Users\Grace\Documents\Code\USRO 2025W\My Code\DS06_data\H2_diffusion\Correlation_Plots\H2_good_mixed.png")
            plt.close()

def diffusion_mixed(datasets):
    datasets = datasets
    for dataset in datasets:
        if dataset == 'H2_mixed':
            x = pd.concat([H2_bad_x,H2_good_x])
            y = pd.concat([H2_bad_y_D,H2_good_y_D])
            train_x,test_x,train_y,test_y = train_test_split(x,y,train_size=0.80,test_size=0.20)

            y_pred_D,test_y_D,srcc_D,r2_D,rmse_D,mae_D = ETR_dif(
                train_x.drop(columns=['MOF']),
                test_x.drop(columns=['MOF']),
                train_y,test_y
            )

            with open(output,'a') as results:
                results.write('\n Metrics for H2 Diffusion. (mixed_mixed) \n')
                results.write('\n Number of MOFs used for training (mixed): ' + str(len(train_x)))
                results.write('\n List of MOFs used for training (mixed): \n' + train_x['MOF'].to_string())
                results.write('\n Number of MOFs used for testing (mixed): ' + str(len(test_x)))
                results.write('\n List of MOFs used for testing (mixed): \n' + test_x['MOF'].to_string())
                results.write('\n Metrics: (Train = Mixed, Test = Mixed, Split = 80 (train), 20 (test)) \n')
                results.write(f'SRCC: {srcc_D}\n')
                results.write(f'R² Score: {r2_D}\n')
                results.write(f'RMSE: {rmse_D}\n')
                results.write(f'MAE: {mae_D}\n')
                results.write('-' * 50 + '\n')
            
            # plotting the correlation graph
            fig = plt.figure(figsize=(7,7),dpi=100)
            plt.rc('axes',axisbelow=True)
            plt.rcParams['font.size'] = 12
            plt.rcParams['font.family'] = "sans-serif"
            plt.grid(color='lightgrey',linestyle='-')
            plt.ylim(ymin=0,ymax=0.0070)
            plt.xlim(xmin=0,xmax=0.0070)
            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8)

            y = np.arange(0,0.01,0.0070)
            x = np.arange(0,0.01,0.0070)
            plt.plot(x,y,color='royalblue',linestyle='--',alpha=0.5)
            plt.scatter(y_pred_D,test_y_D,color='navy')

            plt.xlabel('Predicted H2 Diffusivity (cm2/s)')
            plt.ylabel('Real H2 Diffusivity (cm2/s)')
            plt.title('Predicted vs Real H2 Diffusivities. Train=Mixed, Test=Mixed',y=1.03)
            text = f'SRCC: {srcc_D:.4f} \n R2: {r2_D:.4f} \n RMSE: {rmse_D:.5f} \n MAE: {mae_D:.5f}'
            plt.text(x=0.004,y=0.0005,s=text)
            plt.savefig(r"C:\Users\Grace\Documents\Code\USRO 2025W\My Code\DS06_data\H2_diffusion\Correlation_Plots\H2_mixed_mixed.png")
            plt.close()
        
        elif dataset == 'H2_good':
            train_good_x,test_good_x,train_good_y,test_good_y = train_test_split(H2_good_x,H2_good_y_D,train_size=0.80,test_size=0.20)
            bad_80 = H2_bad.sample(frac=0.80)
            train_bad_x = bad_80.drop(columns=['U_H2 (mol/kg)','D_H2 (cm2/s)'])
            train_bad_y = bad_80['D_H2 (cm2/s)']
            train_x = pd.concat([train_good_x,train_bad_x])
            train_y = pd.concat([train_good_y,train_bad_y])

            y_pred_D,test_y_D,srcc_D,r2_D,rmse_D,mae_D = ETR_dif(
                train_x.drop(columns=['MOF']),
                test_good_x.drop(columns=['MOF']),
                train_y,test_good_y
            )

            with open(output,'a') as results:
                results.write('\n Metrics for H2 Diffusion. (mixed/good) \n')
                results.write('\n Number of MOFs used for training (mixed): ' + str(len(train_x)))
                results.write('\n List of MOFs used for training (mixed): \n' + train_x['MOF'].to_string())
                results.write('\n Number of MOFs used for testing (good): ' + str(len(test_good_x)))
                results.write('\n List of MOFs used for testing (good): \n' + test_good_x['MOF'].to_string())
                results.write('\n Metrics: (Train = Mixed, Test = Good, Split = 80 good + 80 bad (train), 20 good (test)) \n')
                results.write(f'SRCC: {srcc_D}\n')
                results.write(f'R² Score: {r2_D}\n')
                results.write(f'RMSE: {rmse_D}\n')
                results.write(f'MAE: {mae_D}\n')
                results.write('-' * 50 + '\n')
            
            # plotting the correlation graph
            fig = plt.figure(figsize=(7,7),dpi=100)
            plt.rc('axes',axisbelow=True)
            plt.rcParams['font.size'] = 12
            plt.rcParams['font.family'] = "sans-serif"
            plt.grid(color='lightgrey',linestyle='-')
            plt.ylim(ymin=0,ymax=0.0070)
            plt.xlim(xmin=0,xmax=0.0070)
            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8)

            y = np.arange(0,0.01,0.0070)
            x = np.arange(0,0.01,0.0070)
            plt.plot(x,y,color='palevioletred',linestyle='--',alpha=0.5)
            plt.scatter(y_pred_D,test_y_D,color='mediumvioletred')

            plt.xlabel('Predicted H2 Diffusivity (cm2/s)')
            plt.ylabel('Real H2 Diffusivity (cm2/s)')
            plt.title('Predicted vs Real H2 Diffusivities. Train=Mixed, Test=Good',y=1.03)
            text = f'SRCC: {srcc_D:.4f} \n R2: {r2_D:.4f} \n RMSE: {rmse_D:.5f} \n MAE: {mae_D:.5f}'
            plt.text(x=0.004,y=0.0005,s=text)
            plt.savefig(r"C:\Users\Grace\Documents\Code\USRO 2025W\My Code\DS06_data\H2_diffusion\Correlation_Plots\H2_mixed_good.png")
            plt.close()
        
        elif dataset == 'H2_bad':
            train_bad_x,test_bad_x,train_bad_y,test_bad_y = train_test_split(H2_bad_x,H2_bad_y_D,train_size=0.80,test_size=0.20)
            good_80 = H2_good.sample(frac=0.80)
            train_good_x = good_80.drop(columns=['U_H2 (mol/kg)','D_H2 (cm2/s)'])
            train_good_y = good_80['D_H2 (cm2/s)']
            train_x = pd.concat([train_good_x,train_bad_x])
            train_y = pd.concat([train_good_y,train_bad_y])

            y_pred_D,test_y_D,srcc_D,r2_D,rmse_D,mae_D = ETR_dif(
                train_x.drop(columns=['MOF']),
                test_bad_x.drop(columns=['MOF']),
                train_y,test_bad_y
            )

            with open(output,'a') as results:
                results.write('\n Metrics for H2 Diffusion. (mixed/bad) \n')
                results.write('\n Number of MOFs used for training (mixed): ' + str(len(train_x)))
                results.write('\n List of MOFs used for training (mixed): \n' + train_x['MOF'].to_string())
                results.write('\n Number of MOFs used for testing (bad): ' + str(len(test_bad_x)))
                results.write('\n List of MOFs used for testing (bad): \n' + test_bad_x['MOF'].to_string())
                results.write('\n Metrics: (Train = Mixed, Test = Bad, Split = 80 good + 80 bad (train), 20 bad (test)) \n')
                results.write(f'SRCC: {srcc_D}\n')
                results.write(f'R² Score: {r2_D}\n')
                results.write(f'RMSE: {rmse_D}\n')
                results.write(f'MAE: {mae_D}\n')
                results.write('-' * 50 + '\n')
            
            # plotting the correlation graph
            fig = plt.figure(figsize=(7,7),dpi=100)
            plt.rc('axes',axisbelow=True)
            plt.rcParams['font.size'] = 12
            plt.rcParams['font.family'] = "sans-serif"
            plt.grid(color='lightgrey',linestyle='-')
            plt.ylim(ymin=0,ymax=0.0070)
            plt.xlim(xmin=0,xmax=0.0070)
            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8)

            y = np.arange(0,0.01,0.0070)
            x = np.arange(0,0.01,0.0070)
            plt.plot(x,y,color='darkkhaki',linestyle='--',alpha=0.5)
            plt.scatter(y_pred_D,test_y_D,color='darkgoldenrod')

            plt.xlabel('Predicted H2 Diffusivity (cm2/s)')
            plt.ylabel('Real H2 Diffusivity (cm2/s)')
            plt.title('Predicted vs Real H2 Diffusivities. Train=Mixed, Test=Bad',y=1.03)
            text = f'SRCC: {srcc_D:.4f} \n R2: {r2_D:.4f} \n RMSE: {rmse_D:.5f} \n MAE: {mae_D:.5f}'
            plt.text(x=0.004,y=0.0005,s=text)
            plt.savefig(r"C:\Users\Grace\Documents\Code\USRO 2025W\My Code\DS06_data\H2_diffusion\Correlation_Plots\H2_mixed_bad.png")
            plt.close()

def uptake_bad(datasets):
    datasets = datasets
    for dataset in datasets:
        if dataset == 'H2_bad':
            # Data Splitting and Training
            train_x,test_x,train_y,test_y = train_test_split(H2_bad_x,H2_bad_y_A,train_size = 0.80,test_size = 0.20)

            y_pred_A,test_y_A,srcc_A,r2_A,rmse_A,mae_A = ETR_ads(
                train_x.drop(columns=['MOF']),
                test_x.drop(columns=['MOF']),
                train_y,test_y
                )
            
            #Reporting metrics
            
            with open(output,'a') as results:
                results.write('\n Metrics for H2 Adsorption. (bad/bad) \n')
                results.write('\n Number of MOFs used for training (bad): ' + str(len(train_x)))
                results.write('\n List of MOFs used for training (bad): \n' + train_x['MOF'].to_string())
                results.write('\n Number of MOFs used for testing (bad): ' + str(len(test_x)))
                results.write('\n List of MOFs used for testing (bad): \n' + test_x['MOF'].to_string())
                results.write('\n Metrics: (Train = Bad, Test = Bad, Split = 80/20) \n')
                results.write(f'SRCC: {srcc_A}\n')
                results.write(f'R² Score: {r2_A}\n')
                results.write(f'RMSE: {rmse_A}\n')
                results.write(f'MAE: {mae_A}\n')
                results.write('-' * 50 + '\n')
            
            # plotting the correlation graph
            fig = plt.figure(figsize=(7,7),dpi=100)
            plt.rc('axes',axisbelow=True)
            plt.rcParams['font.size'] = 12
            plt.rcParams['font.family'] = "sans-serif"
            plt.grid(color='lightgrey',linestyle='-')
            plt.ylim(ymin=0,ymax=0.25)
            plt.xlim(xmin=0,xmax=0.25)
            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8)

            y = np.arange(0,0.35,0.05)
            x = np.arange(0,0.35,0.05)
            plt.plot(x,y,color='lightgreen',linestyle='--',alpha=0.5)
            plt.scatter(y_pred_A,test_y_A,color='forestgreen')

            plt.xlabel('Predicted H2 Adsorption (mol/kg)')
            plt.ylabel('Real H2 Adsorption (mol/kg)')
            plt.title('Predicted vs Real H2 Adsorption. Train=Bad, Test=Bad',y=1.03)
            text = f'SRCC: {srcc_A:.4f} \n R2: {r2_A:.4f} \n RMSE: {rmse_A:.5f} \n MAE: {mae_A:.5f}'
            plt.text(x=0.20,y=0.06,s=text)

            plt.savefig(r"C:\Users\Grace\Documents\Code\USRO 2025W\My Code\DS06_data\H2_adsorption\Correlation_Plots\H2_bad_bad.png")
            plt.close()


        elif dataset == 'H2_good':
            y_pred_A,test_y_A,srcc_A,r2_A,rmse_A,mae_A = ETR_ads(
                H2_bad_x.drop(columns=['MOF']),
                H2_good_x.drop(columns=['MOF']),
                H2_bad_y_A,H2_good_y_A
                )

            with open(output,'a') as results:
                results.write('\n Metrics for H2 Adsorption. (bad/good) \n')
                results.write('\n Number of MOFs used for training(bad): ' + str(len(H2_bad_x)))
                results.write('\n List of MOFs used for training (bad): \n' + H2_bad_x['MOF'].to_string())
                results.write('\n Number of MOFs used for testing (good): '+ str(len(H2_good_y_A)))
                results.write('\n List of MOFs used for testing (good): \n' + H2_good_x['MOF'].to_string())
                results.write('\n Metrics: (Train = Bad, Test = Good, No split.) \n')
                results.write(f'SRCC: {srcc_A}\n')
                results.write(f'R² Score: {r2_A}\n')
                results.write(f'RMSE: {rmse_A}\n')
                results.write(f'MAE: {mae_A}\n')
                results.write('-' * 50 + '\n')
            
            # plotting the correlation graph
            fig = plt.figure(figsize=(7,7),dpi=100)
            plt.rc('axes',axisbelow=True)
            plt.rcParams['font.size'] = 12
            plt.rcParams['font.family'] = "sans-serif"
            plt.grid(color='lightgrey',linestyle='-')
            plt.ylim(ymin=0,ymax=0.25)
            plt.xlim(xmin=0,xmax=0.25)
            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8)

            y = np.arange(0,0.35,0.05)
            x = np.arange(0,0.35,0.05)
            plt.plot(x,y,color='indianred',linestyle='--',alpha=0.5)
            plt.scatter(y_pred_A,test_y_A,color='firebrick')

            plt.xlabel('Predicted H2 Adsorption (mol/kg)')
            plt.ylabel('Real H2 Adsorption (mol/kg)')
            plt.title('Predicted vs Real H2 Adsorptions. Train=Bad, Test=Good',y=1.03)
            text = f'SRCC: {srcc_A:.4f} \n R2: {r2_A:.4f} \n RMSE: {rmse_A:.5f} \n MAE: {mae_A:.5f}'
            plt.text(x=0.20,y=0.06,s=text)
            plt.savefig(r"C:\Users\Grace\Documents\Code\USRO 2025W\My Code\DS06_data\H2_adsorption\Correlation_Plots\H2_bad_good.png")
            plt.close()


        elif dataset == 'H2_mixed':
            train_x,test_bad_x,train_y,test_bad_y_A = train_test_split(H2_bad_x,H2_bad_y_A,train_size=0.80,test_size=0.20)

            test_good_x = H2_good_x.sample(n=len(test_bad_x))
            test_good_y_A = H2_good_y_A.sample(n=len(test_bad_y_A))
            test_x = pd.concat([test_bad_x,test_good_x])
            test_y_A = pd.concat([test_bad_y_A,test_good_y_A])

            y_pred_A,test_y_A,srcc_A,r2_A,rmse_A,mae_A = ETR_ads(
                train_x.drop(columns=['MOF']),
                test_x.drop(columns=['MOF']),
                train_y,test_y_A
                )

            with open(output,'a') as results:
                results.write('\n Metrics for H2 Adsorption. (bad/mixed) \n')
                results.write('\n Number of MOFs used for training (bad): ' + str(len(train_x)))
                results.write('\n List of MOFs used for training (bad): \n' + train_x['MOF'].to_string())
                results.write('\n Number of MOFs used for testing (mixed): ' + str(len(test_x)))
                results.write('\n List of MOFs used for testing (mixed): \n' + test_x['MOF'].to_string())
                results.write('\n Metrics: (Train = Bad, Test = Mixed, Split = 80 (train), 20(n) bad + n good (test)) \n')
                results.write(f'SRCC: {srcc_A}\n')
                results.write(f'R² Score: {r2_A}\n')
                results.write(f'RMSE: {rmse_A}\n')
                results.write(f'MAE: {mae_A}\n')
                results.write('-' * 50 + '\n')
            
            # plotting the correlation graph
            fig = plt.figure(figsize=(7,7),dpi=100)
            plt.rc('axes',axisbelow=True)
            plt.rcParams['font.size'] = 12
            plt.rcParams['font.family'] = "sans-serif"
            plt.grid(color='lightgrey',linestyle='-')
            plt.ylim(ymin=0,ymax=0.25)
            plt.xlim(xmin=0,xmax=0.25)
            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8)

            y = np.arange(0,0.35,0.05)
            x = np.arange(0,0.35,0.05)
            plt.plot(x,y,color='sandybrown',linestyle='--',alpha=0.5)
            plt.scatter(y_pred_A,test_y_A,color='darkorange')

            plt.xlabel('Predicted H2 Adsorption (mol/kg)')
            plt.ylabel('Real H2 Adsorption (mol/kg)')
            plt.title('Predicted vs Real H2 Adsorptions. Train=Bad, Test=Mixed',y=1.03)
            text = f'SRCC: {srcc_A:.4f} \n R2: {r2_A:.4f} \n RMSE: {rmse_A:.5f} \n MAE: {mae_A:.5f}'
            plt.text(x=0.20,y=0.06,s=text)
            plt.savefig(r"C:\Users\Grace\Documents\Code\USRO 2025W\My Code\DS06_data\H2_adsorption\Correlation_Plots\H2_bad_mixed.png")
            plt.close()

def uptake_good(datasets):
    datasets = datasets
    for dataset in datasets:
        if dataset == 'H2_good':
            # Data Splitting and Training
            train_x,test_x,train_y,test_y = train_test_split(H2_good_x,H2_good_y_A,train_size = 0.80,test_size = 0.20)

            y_pred_A,test_y_A,srcc_A,r2_A,rmse_A,mae_A = ETR_ads(
                train_x.drop(columns=['MOF']),
                test_x.drop(columns=['MOF']),
                train_y,test_y
                )
            
            #Reporting metrics
            
            with open(output,'a') as results:
                results.write('\n Metrics for H2 Adsorption. (good/good) \n')
                results.write('\n Number of MOFs used for training (good): ' + str(len(train_x)))
                results.write('\n List of MOFs used for training (good): \n' + train_x['MOF'].to_string())
                results.write('\n Number of MOFs used for testing (good): ' + str(len(test_x)))
                results.write('\n List of MOFs used for testing (good): \n' + test_x['MOF'].to_string())
                results.write('\n Metrics: (Train = Good, Test = Good, Split = 80/20) \n')
                results.write(f'SRCC: {srcc_A}\n')
                results.write(f'R² Score: {r2_A}\n')
                results.write(f'RMSE: {rmse_A}\n')
                results.write(f'MAE: {mae_A}\n')
                results.write('-' * 50 + '\n')
            
            # plotting the correlation graph
            fig = plt.figure(figsize=(7,7),dpi=100)
            plt.rc('axes',axisbelow=True)
            plt.rcParams['font.size'] = 12
            plt.rcParams['font.family'] = "sans-serif"
            plt.grid(color='lightgrey',linestyle='-')
            plt.ylim(ymin=0,ymax=0.25)
            plt.xlim(xmin=0,xmax=0.25)
            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8)

            y = np.arange(0,0.35,0.05)
            x = np.arange(0,0.35,0.05)
            plt.plot(x,y,color='cadetblue',linestyle='--',alpha=0.5)
            plt.scatter(y_pred_A,test_y_A,color='teal')

            plt.xlabel('Predicted H2 Adsorption (mol/kg)')
            plt.ylabel('Real H2 Adsorption (mol/kg)')
            plt.title('Predicted vs Real H2 Adsorptions. Train=Good, Test=Good',y=1.03)
            text = f'SRCC: {srcc_A:.4f} \n R2: {r2_A:.4f} \n RMSE: {rmse_A:.5f} \n MAE: {mae_A:.5f}'
            plt.text(x=0.20,y=0.06,s=text)
            plt.savefig(r"C:\Users\Grace\Documents\Code\USRO 2025W\My Code\DS06_data\H2_adsorption\Correlation_Plots\H2_good_good.png")
            plt.close()
        
        elif dataset =='H2_bad':
            y_pred_A,test_y_A,srcc_A,r2_A,rmse_A,mae_A = ETR_ads(
                H2_good_x.drop(columns=['MOF']),
                H2_bad_x.drop(columns=['MOF']),
                H2_good_y_A,H2_bad_y_A
                )

            with open(output,'a') as results:
                results.write('\n Metrics for H2 Adsorption. (good/bad) \n')
                results.write('\n Number of MOFs used for training(good): ' + str(len(H2_good_x)))
                results.write('\n List of MOFs used for training (good): \n' + H2_good_x['MOF'].to_string())
                results.write('\n Number of MOFs used for testing (bad): '+ str(len(H2_bad_x)))
                results.write('\n List of MOFs used for testing (bad): \n' + H2_bad_x['MOF'].to_string())
                results.write('\n Metrics: (Train = Good, Test = Bad, No split.) \n')
                results.write(f'SRCC: {srcc_A}\n')
                results.write(f'R² Score: {r2_A}\n')
                results.write(f'RMSE: {rmse_A}\n')
                results.write(f'MAE: {mae_A}\n')
                results.write('-' * 50 + '\n')
            
            # plotting the correlation graph
            fig = plt.figure(figsize=(7,7),dpi=100)
            plt.rc('axes',axisbelow=True)
            plt.rcParams['font.size'] = 12
            plt.rcParams['font.family'] = "sans-serif"
            plt.grid(color='lightgrey',linestyle='-')
            plt.ylim(ymin=0,ymax=0.25)
            plt.xlim(xmin=0,xmax=0.25)
            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8)

            y = np.arange(0,0.35,0.05)
            x = np.arange(0,0.35,0.05)
            plt.plot(x,y,color='slateblue',linestyle='--',alpha=0.5)
            plt.scatter(y_pred_A,test_y_A,color='darkslateblue')

            plt.xlabel('Predicted H2 Adsorption (mol/kg)')
            plt.ylabel('Real H2 Adsorption (mol/kg)')
            plt.title('Predicted vs Real H2 Adsorptions. Train=Good, Test=Bad',y=1.03)
            text = f'SRCC: {srcc_A:.4f} \n R2: {r2_A:.4f} \n RMSE: {rmse_A:.5f} \n MAE: {mae_A:.5f}'
            plt.text(x=0.20,y=0.06,s=text)
            plt.savefig(r"C:\Users\Grace\Documents\Code\USRO 2025W\My Code\DS06_data\H2_adsorption\Correlation_Plots\H2_good_bad.png")
            plt.close()
        
        elif dataset =='H2_mixed':
            train_x,test_good_x,train_y,test_good_y_A = train_test_split(H2_good_x,H2_good_y_A,train_size=0.80,test_size=0.20)

            test_bad_x = H2_bad_x.sample(n=len(test_good_x))
            test_bad_y_A = H2_bad_y_A.sample(n=len(test_good_y_A))
            test_x = pd.concat([test_bad_x,test_good_x])
            test_y_A = pd.concat([test_bad_y_A,test_good_y_A])

            y_pred_A,test_y_A,srcc_A,r2_A,rmse_A,mae_A = ETR_ads(
                train_x.drop(columns=['MOF']),
                test_x.drop(columns=['MOF']),
                train_y,test_y_A
                )

            with open(output,'a') as results:
                results.write('\n Metrics for H2 Adsorption. (good/mixed) \n')
                results.write('\n Number of MOFs used for training (good): ' + str(len(train_x)))
                results.write('\n List of MOFs used for training (good): \n' + train_x['MOF'].to_string())
                results.write('\n Number of MOFs used for testing (mixed): ' + str(len(test_x)))
                results.write('\n List of MOFs used for testing (mixed): \n' + test_x['MOF'].to_string())
                results.write('\n Metrics: (Train = Good, Test = Mixed, Split = 80 (train), 20(n) good + n bad (test)) \n')
                results.write(f'SRCC: {srcc_A}\n')
                results.write(f'R² Score: {r2_A}\n')
                results.write(f'RMSE: {rmse_A}\n')
                results.write(f'MAE: {mae_A}\n')
                results.write('-' * 50 + '\n')
            
            # plotting the correlation graph
            fig = plt.figure(figsize=(7,7),dpi=100)
            plt.rc('axes',axisbelow=True)
            plt.rcParams['font.size'] = 12
            plt.rcParams['font.family'] = "sans-serif"
            plt.grid(color='lightgrey',linestyle='-')
            plt.ylim(ymin=0,ymax=0.25)
            plt.xlim(xmin=0,xmax=0.25)
            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8)

            y = np.arange(0,0.35,0.05)
            x = np.arange(0,0.35,0.05)
            plt.plot(x,y,color='plum',linestyle='--',alpha=0.5)
            plt.scatter(y_pred_A,test_y_A,color='mediumvioletred')

            plt.xlabel('Predicted H2 Adsorption (mol/kg)')
            plt.ylabel('Real H2 Adsorption (mol/kg)')
            plt.title('Predicted vs Real H2 Adsorptions. Train=Good, Test=Mixed',y=1.03)
            text = f'SRCC: {srcc_A:.4f} \n R2: {r2_A:.4f} \n RMSE: {rmse_A:.5f} \n MAE: {mae_A:.5f}'
            plt.text(x=0.25,y=0.06,s=text)
            plt.savefig(r"C:\Users\Grace\Documents\Code\USRO 2025W\My Code\DS06_data\H2_adsorption\Correlation_Plots\H2_good_mixed.png")
            plt.close()

def uptake_mixed(datasets):
    datasets = datasets
    for dataset in datasets:
        if dataset == 'H2_mixed':
            x = pd.concat([H2_bad_x,H2_good_x])
            y = pd.concat([H2_bad_y_A,H2_good_y_A])
            train_x,test_x,train_y,test_y = train_test_split(x,y,train_size=0.80,test_size=0.20)

            y_pred_A,test_y_A,srcc_A,r2_A,rmse_A,mae_A = ETR_ads(
                train_x.drop(columns=['MOF']),
                test_x.drop(columns=['MOF']),
                train_y,test_y
            )

            with open(output,'a') as results:
                results.write('\n Metrics for H2 Adsorption. (mixed_mixed) \n')
                results.write('\n Number of MOFs used for training (mixed): ' + str(len(train_x)))
                results.write('\n List of MOFs used for training (mixed): \n' + train_x['MOF'].to_string())
                results.write('\n Number of MOFs used for testing (mixed): ' + str(len(test_x)))
                results.write('\n List of MOFs used for testing (mixed): \n' + test_x['MOF'].to_string())
                results.write('\n Metrics: (Train = Mixed, Test = Mixed, Split = 80 (train), 20 (test)) \n')
                results.write(f'SRCC: {srcc_A}\n')
                results.write(f'R² Score: {r2_A}\n')
                results.write(f'RMSE: {rmse_A}\n')
                results.write(f'MAE: {mae_A}\n')
                results.write('-' * 50 + '\n')
            
            # plotting the correlation graph
            fig = plt.figure(figsize=(7,7),dpi=100)
            plt.rc('axes',axisbelow=True)
            plt.rcParams['font.size'] = 12
            plt.rcParams['font.family'] = "sans-serif"
            plt.grid(color='lightgrey',linestyle='-')
            plt.ylim(ymin=0,ymax=0.25)
            plt.xlim(xmin=0,xmax=0.25)
            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8)

            y = np.arange(0,0.35,0.05)
            x = np.arange(0,0.35,0.05)
            plt.plot(x,y,color='royalblue',linestyle='--',alpha=0.5)
            plt.scatter(y_pred_A,test_y_A,color='navy')

            plt.xlabel('Predicted H2 Adsorption (mol/kg)')
            plt.ylabel('Real H2 Adsorption (mol/kg)')
            plt.title('Predicted vs Real H2 Adsorptions. Train=Mixed, Test=Mixed',y=1.03)
            text = f'SRCC: {srcc_A:.4f} \n R2: {r2_A:.4f} \n RMSE: {rmse_A:.5f} \n MAE: {mae_A:.5f}'
            plt.text(x=0.20,y=0.06,s=text)
            plt.savefig(r"C:\Users\Grace\Documents\Code\USRO 2025W\My Code\DS06_data\H2_adsorption\Correlation_Plots\H2_mixed_mixed.png")
            plt.close()
        
        elif dataset == 'H2_good':
            train_good_x,test_good_x,train_good_y,test_good_y = train_test_split(H2_good_x,H2_good_y_A,train_size=0.80,test_size=0.20)
            bad_80 = H2_bad.sample(frac=0.80)
            train_bad_x = bad_80.drop(columns=['U_H2 (mol/kg)','D_H2 (cm2/s)'])
            train_bad_y = bad_80['U_H2 (mol/kg)']
            train_x = pd.concat([train_good_x,train_bad_x])
            train_y = pd.concat([train_good_y,train_bad_y])

            y_pred_A,test_y_A,srcc_A,r2_A,rmse_A,mae_A = ETR_ads(
                train_x.drop(columns=['MOF']),
                test_good_x.drop(columns=['MOF']),
                train_y,test_good_y
            )

            with open(output,'a') as results:
                results.write('\n Metrics for H2 Adsorption. (mixed/good) \n')
                results.write('\n Number of MOFs used for training (mixed): ' + str(len(train_x)))
                results.write('\n List of MOFs used for training (mixed): \n' + train_x['MOF'].to_string())
                results.write('\n Number of MOFs used for testing (good): ' + str(len(test_good_x)))
                results.write('\n List of MOFs used for testing (good): \n' + test_good_x['MOF'].to_string())
                results.write('\n Metrics: (Train = Mixed, Test = Good, Split = 80 good + 80 bad (train), 20 good (test)) \n')
                results.write(f'SRCC: {srcc_A}\n')
                results.write(f'R² Score: {r2_A}\n')
                results.write(f'RMSE: {rmse_A}\n')
                results.write(f'MAE: {mae_A}\n')
                results.write('-' * 50 + '\n')
            
            # plotting the correlation graph
            fig = plt.figure(figsize=(7,7),dpi=100)
            plt.rc('axes',axisbelow=True)
            plt.rcParams['font.size'] = 12
            plt.rcParams['font.family'] = "sans-serif"
            plt.grid(color='lightgrey',linestyle='-')
            plt.ylim(ymin=0,ymax=0.25)
            plt.xlim(xmin=0,xmax=0.25)
            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8)

            y = np.arange(0,0.35,0.05)
            x = np.arange(0,0.35,0.05)
            plt.plot(x,y,color='palevioletred',linestyle='--',alpha=0.5)
            plt.scatter(y_pred_A,test_y_A,color='mediumvioletred')

            plt.xlabel('Predicted H2 Adsorption (mol/kg)')
            plt.ylabel('Real H2 Adsorption (mol/kg)')
            plt.title('Predicted vs Real H2 Adsorptions. Train=Mixed, Test=Good',y=1.03)
            text = f'SRCC: {srcc_A:.4f} \n R2: {r2_A:.4f} \n RMSE: {rmse_A:.5f} \n MAE: {mae_A:.5f}'
            plt.text(x=0.20,y=0.06,s=text)
            plt.savefig(r"C:\Users\Grace\Documents\Code\USRO 2025W\My Code\DS06_data\H2_adsorption\Correlation_Plots\H2_mixed_good.png")
            plt.close()
        
        elif dataset == 'H2_bad':
            train_bad_x,test_bad_x,train_bad_y,test_bad_y = train_test_split(H2_bad_x,H2_bad_y_A,train_size=0.80,test_size=0.20)
            good_80 = H2_good.sample(frac=0.80)
            train_good_x = good_80.drop(columns=['U_H2 (mol/kg)','D_H2 (cm2/s)'])
            train_good_y = good_80['U_H2 (mol/kg)']
            train_x = pd.concat([train_good_x,train_bad_x])
            train_y = pd.concat([train_good_y,train_bad_y])

            y_pred_A,test_y_A,srcc_A,r2_A,rmse_A,mae_A = ETR_ads(
                train_x.drop(columns=['MOF']),
                test_bad_x.drop(columns=['MOF']),
                train_y,test_bad_y
            )

            with open(output,'a') as results:
                results.write('\n Metrics for H2 Adsorption. (mixed/bad) \n')
                results.write('\n Number of MOFs used for training (mixed): ' + str(len(train_x)))
                results.write('\n List of MOFs used for training (mixed): \n' + train_x['MOF'].to_string())
                results.write('\n Number of MOFs used for testing (bad): ' + str(len(test_bad_x)))
                results.write('\n List of MOFs used for testing (bad): \n' + test_bad_x['MOF'].to_string())
                results.write('\n Metrics: (Train = Mixed, Test = Bad, Split = 80 good + 80 bad (train), 20 bad (test)) \n')
                results.write(f'SRCC: {srcc_A}\n')
                results.write(f'R² Score: {r2_A}\n')
                results.write(f'RMSE: {rmse_A}\n')
                results.write(f'MAE: {mae_A}\n')
                results.write('-' * 50 + '\n')
            
            # plotting the correlation graph
            fig = plt.figure(figsize=(7,7),dpi=100)
            plt.rc('axes',axisbelow=True)
            plt.rcParams['font.size'] = 12
            plt.rcParams['font.family'] = "sans-serif"
            plt.grid(color='lightgrey',linestyle='-')
            plt.ylim(ymin=0,ymax=0.25)
            plt.xlim(xmin=0,xmax=0.25)
            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8)

            y = np.arange(0,0.35,0.05)
            x = np.arange(0,0.35,0.05)
            plt.plot(x,y,color='darkkhaki',linestyle='--',alpha=0.5)
            plt.scatter(y_pred_A,test_y_A,color='darkgoldenrod')

            plt.xlabel('Predicted H2 Adsorption (mol/kg)')
            plt.ylabel('Real H2 Adsorption (mol/kg)')
            plt.title('Predicted vs Real H2 Adsorption. Train=Mixed, Test=Bad',y=1.03)
            text = f'SRCC: {srcc_A:.4f} \n R2: {r2_A:.4f} \n RMSE: {rmse_A:.5f} \n MAE: {mae_A:.5f}'
            plt.text(x=0.20,y=0.06,s=text)
            plt.savefig(r"C:\Users\Grace\Documents\Code\USRO 2025W\My Code\DS06_data\H2_adsorption\Correlation_Plots\H2_mixed_bad.png")
            plt.close()
