# This is a replication of the Random Forest Regressor model used in the paper DS06.
# This model predicts CH4 diffusion, it uses R2, MAE,SRCC, and RMSE as evaluators. 
# All molecular descriptors are used in the training 
# The paper itself did not specify whether or not features were preprocessed,
# I used Standard Scaler.

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
from decimal import Decimal

# Import data and preprocessing - Either by command prompt or hardcode
''' 
args = argv # enter data fp
fp = args[1]
'''
fp =r"C:\Users\Grace\Documents\Code\USRO 2025W\Raw_Data\DS06\rawdata\Methane.csv" # paste fp here
ASR_bad = r"C:\Users\Grace\Documents\Code\USRO 2025W\Good_Bad_MOF_splits\DS06\Bad\DS06_ASR_bad_output_H2.txt"
FSR_bad = r"C:\Users\Grace\Documents\Code\USRO 2025W\Good_Bad_MOF_splits\DS06\Bad\DS06_FSR_bad_output_H2.txt"
ASR_good = r"C:\Users\Grace\Documents\Code\USRO 2025W\Good_Bad_MOF_splits\DS06\Good\DS06_ASR_good_output_H2.txt"
FSR_good = r"C:\Users\Grace\Documents\Code\USRO 2025W\Good_Bad_MOF_splits\DS06\Good\DS06_FSR_good_output_H2.txt"

# Good/Bad MOF definition and splitting
asr_bad = open(ASR_bad,'r').read().splitlines()
fsr_bad = open(FSR_bad,'r').read().splitlines()
bad_names = set(asr_bad+fsr_bad)
asr_good = open(ASR_good,'r').read().splitlines()
fsr_good = open(FSR_good,'r').read().splitlines()
good_names = set(asr_good+fsr_good)

data = pd.read_csv(fp)

good = data[data['MOF name'].isin(good_names)]
bad = data[data['MOF name'].isin(bad_names)]
good_x = good.drop(columns = ['U_CH4 (mol/kg)','D_CH4 (cm2/s)'])
bad_x = bad.drop(columns = ['U_CH4 (mol/kg)','D_CH4 (cm2/s)'])
good_y = good['D_CH4 (cm2/s)']
bad_y = bad['D_CH4 (cm2/s)']

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


# model training 
def RFR_CH4(train_x,test_x,train_y,test_y):
    train_x = train_x
    test_x = test_x 
    train_y = train_y 
    test_y = test_y 

    scaler = StandardScaler().fit(train_x)
    train_x = scaler.transform(train_x)
    test_x = scaler.transform(test_x)

    model = RandomForestRegressor(
        n_estimators = 1000,
        max_depth = 50,
        min_samples_split = 2,
        min_samples_leaf = 1,
        bootstrap = False,
        max_features = 'sqrt',
        random_state = 42
    )

    y_pred = model.fit(train_x,train_y).predict(test_x)
    r2 = r2_score(y_pred,test_y)
    rmse = np.sqrt(mean_squared_error(y_pred,test_y))
    mae = mean_absolute_error(y_pred,test_y)
    srcc = get_SRCC(y_pred,test_y)

    return r2,rmse,mae,srcc,y_pred

tests = ['CH4_bad','CH4_good','CH4_mixed']
output_fp = r"C:\Users\Grace\Documents\Code\USRO 2025W\My Code\DS06_data\CH4_diffusion\Train_mixed.txt"

def train_bad(datasets):
    for dataset in datasets:
        if dataset == 'CH4_bad':
            train_x,test_x,train_y,test_y = train_test_split(bad_x,bad_y,train_size=0.80,test_size=0.20,random_state = 42)
            
            r2,rmse,mae,srcc,y_pred = RFR_CH4(train_x.drop(columns=['MOF name']),test_x.drop(columns=['MOF name']),train_y,test_y)

            # Output Metrics
            with open(output_fp,'a') as results:
                results.write('\n CH4 Diffusivity Data. Train = Bad, Test = Bad, Split = 80/20')
                results.write('\n Number of MOFs used in Training (Bad): ' + str(len(train_x)))
                results.write('\n List of MOFs used in Training (Bad): \n' + train_x['MOF name'].to_string())
                results.write('='*50)
                results.write('\n Number of MOFs used in Testing (Bad): ' + str(len(test_x)))
                results.write('\n List of MOFs used in Testing (Bad): \n' + test_x['MOF name'].to_string())
                results.write('='*50)
                results.write('\n Metrics of Evaluation: (Bad/Bad)')
                results.write('\n R2 Score: ' + str(r2))
                results.write('\n RMSE: ' + str(rmse))
                results.write('\n MAE: ' + str(mae))
                results.write('\n SRCC: ' + str(srcc))
                results.write('\n' + '=+='*20)
            
            # Correlation Plot
            fig = plt.figure(figsize=(7,7),dpi=100)
            plt.rc('axes',axisbelow=True)
            plt.rcParams['font.size'] = 12
            plt.rcParams['font.family'] = 'sans-serif'
            plt.grid(color = 'lightgrey',linestyle='-')
            plt.ylim(ymin=0,ymax=0.0013)
            plt.xlim(xmin=0,xmax=0.0013)
            plt.yticks(fontsize=8)
            plt.xticks(fontsize=8)

            x = np.arange(0,0.0030,0.0010)
            y = np.arange(0,0.0030,0.0010)
            plt.plot(x,y,color='darkgoldenrod',linestyle='--',alpha=0.50)
            plt.scatter(y_pred,test_y,color='chocolate')

            plt.xlabel('Predicted CH4 Diffusivity (cm2/s)')
            plt.ylabel('Real CH4 Diffusivity (cm2/s)')
            plt.title('Plot of Real vs Predicted CH4 Diffusivity. Train=Bad, Test=Bad')
            text = f'R2: {r2:.4f} \n RMSE: {Decimal(rmse):.2E} \n MAE: {Decimal(mae):.2E} \n SRCC: {srcc:.4f}'
            plt.text(x=0.0009,y=0.00020,s=text)
            plt.savefig(r"C:\Users\Grace\Documents\Code\USRO 2025W\My Code\DS06_data\CH4_diffusion\Correlation_Plots\CH4_bad_bad.png")
            plt.close()

        elif dataset == 'CH4_good':
            train_x = bad_x
            test_x = good_x
            train_y = bad_y
            test_y = good_y

            r2,rmse,mae,srcc,y_pred = RFR_CH4(train_x.drop(columns=['MOF name']),test_x.drop(columns=['MOF name']),train_y,test_y)

            # Output Metrics
            with open(output_fp,'a') as results:
                results.write('\n CH4 Diffusivity Data. Train = Bad, Test = Good, No splits')
                results.write('\n Number of MOFs used in Training (Bad): ' + str(len(train_x)))
                results.write('\n List of MOFs used in Training (Bad): \n' + train_x['MOF name'].to_string())
                results.write('='*50)
                results.write('\n Number of MOFs used in Testing (Good): ' + str(len(test_x)))
                results.write('\n List of MOFs used in Testing (Good): \n' + test_x['MOF name'].to_string())
                results.write('='*50)
                results.write('\n Metrics of Evaluation: (Bad/Good)')
                results.write('\n R2 Score: ' + str(r2))
                results.write('\n RMSE: ' + str(rmse))
                results.write('\n MAE: ' + str(mae))
                results.write('\n SRCC: ' + str(srcc))
                results.write('\n' + '=+='*20)
            
            # Correlation Plot
            fig = plt.figure(figsize=(7,7),dpi=100)
            plt.rc('axes',axisbelow=True)
            plt.rcParams['font.size'] = 12
            plt.rcParams['font.family'] = 'sans-serif'
            plt.grid(color = 'lightgrey',linestyle='-')
            plt.ylim(ymin=0,ymax=0.0013)
            plt.xlim(xmin=0,xmax=0.0013)
            plt.yticks(fontsize=8)
            plt.xticks(fontsize=8)

            x = np.arange(0,0.0030,0.0010)
            y = np.arange(0,0.0030,0.0010)
            plt.plot(x,y,color='indianred',linestyle='--',alpha=0.50)
            plt.scatter(y_pred,test_y,color='firebrick')

            plt.xlabel('Predicted CH4 Diffusivity (cm2/s)')
            plt.ylabel('Real CH4 Diffusivity (cm2/s)')
            plt.title('Plot of Real vs Predicted CH4 Diffusivity. Train=Bad, Test=Good')
            text = f'R2: {r2:.4f} \n RMSE: {Decimal(rmse):.2E} \n MAE: {Decimal(mae):.2E} \n SRCC: {srcc:.4f}'
            plt.text(x=0.0009,y=0.00020,s=text)
            plt.savefig(r"C:\Users\Grace\Documents\Code\USRO 2025W\My Code\DS06_data\CH4_diffusion\Correlation_Plots\CH4_bad_good.png")
            plt.close()

        elif dataset == 'CH4_mixed':
            train_x,test_x_bad,train_y,test_y_bad = train_test_split(bad_x,bad_y,train_size=0.80,test_size=0.20)
            good_20 = good.sample(n=len(test_x_bad))
            test_x_good = good_20.drop(columns=['U_CH4 (mol/kg)','D_CH4 (cm2/s)'])
            test_y_good = good_20['D_CH4 (cm2/s)']

            test_x = pd.concat([test_x_bad,test_x_good])
            test_y = pd.concat([test_y_bad,test_y_good])

            r2,rmse,mae,srcc,y_pred = RFR_CH4(
                train_x.drop(columns=['MOF name']),
                test_x.drop(columns=['MOF name']),
                train_y,test_y)

            # Output Metrics
            with open(output_fp,'a') as results:
                results.write('\n CH4 Diffusivity Data. Train = Bad, Test = Mixed, Split: 80/(20:20)')
                results.write('\n Number of MOFs used in Training (Bad): ' + str(len(train_x)))
                results.write('\n List of MOFs used in Training (Bad): \n' + train_x['MOF name'].to_string())
                results.write('='*50)
                results.write('\n Number of MOFs used in Testing (Mixed): ' + str(len(test_x)))
                results.write('\n List of MOFs used in Testing (Mixed): \n' + test_x['MOF name'].to_string())
                results.write('='*50)
                results.write('\n Metrics of Evaluation: (Bad/Mixed)')
                results.write('\n R2 Score: ' + str(r2))
                results.write('\n RMSE: ' + str(rmse))
                results.write('\n MAE: ' + str(mae))
                results.write('\n SRCC: ' + str(srcc))
                results.write('\n' + '=+='*20)
            
            # Correlation Plot
            fig = plt.figure(figsize=(7,7),dpi=100)
            plt.rc('axes',axisbelow=True)
            plt.rcParams['font.size'] = 12
            plt.rcParams['font.family'] = 'sans-serif'
            plt.grid(color = 'lightgrey',linestyle='-')
            plt.ylim(ymin=0,ymax=0.0013)
            plt.xlim(xmin=0,xmax=0.0013)
            plt.yticks(fontsize=8)
            plt.xticks(fontsize=8)

            x = np.arange(0,0.0030,0.0010)
            y = np.arange(0,0.0030,0.0010)
            plt.plot(x,y,color='sandybrown',linestyle='--',alpha=0.50)
            plt.scatter(y_pred,test_y,color='darkorange')

            plt.xlabel('Predicted CH4 Diffusivity (cm2/s)')
            plt.ylabel('Real CH4 Diffusivity (cm2/s)')
            plt.title('Plot of Real vs Predicted CH4 Diffusivity. Train=Bad, Test=Mixed')
            text = f'R2: {r2:.4f} \n RMSE: {Decimal(rmse):.2E} \n MAE: {Decimal(mae):.2E} \n SRCC: {srcc:.4f}'
            plt.text(x=0.0009,y=0.00020,s=text)
            plt.savefig(r"C:\Users\Grace\Documents\Code\USRO 2025W\My Code\DS06_data\CH4_diffusion\Correlation_Plots\CH4_bad_mixed.png")
            plt.close()

def train_good(datasets):
    for dataset in datasets: 
        if dataset=='CH4_good':
            train_x,test_x,train_y,test_y = train_test_split(good_x,good_y,train_size=0.80,test_size=0.20,random_state=42)

            r2,rmse,mae,srcc,y_pred = RFR_CH4(train_x.drop(columns=['MOF name']),test_x.drop(columns=['MOF name']),train_y,test_y)

            # Output Metrics
            with open(output_fp,'a') as results:
                results.write('\n CH4 Diffusivity Data. Train = Good, Test = Good, Split = 80/20')
                results.write('\n Number of MOFs used in Training (Good): ' + str(len(train_x)))
                results.write('\n List of MOFs used in Training (Good): \n' + train_x['MOF name'].to_string())
                results.write('='*50)
                results.write('\n Number of MOFs used in Testing (Good): ' + str(len(test_x)))
                results.write('\n List of MOFs used in Testing (Good): \n' + test_x['MOF name'].to_string())
                results.write('='*50)
                results.write('\n Metrics of Evaluation: (Good/Good)')
                results.write('\n R2 Score: ' + str(r2))
                results.write('\n RMSE: ' + str(rmse))
                results.write('\n MAE: ' + str(mae))
                results.write('\n SRCC: ' + str(srcc))
                results.write('\n' + '=+='*20)
            
            # Correlation Plot
            fig = plt.figure(figsize=(7,7),dpi=100)
            plt.rc('axes',axisbelow=True)
            plt.rcParams['font.size'] = 12
            plt.rcParams['font.family'] = 'sans-serif'
            plt.grid(color = 'lightgrey',linestyle='-')
            plt.ylim(ymin=0,ymax=0.0013)
            plt.xlim(xmin=0,xmax=0.0013)
            plt.yticks(fontsize=8)
            plt.xticks(fontsize=8)

            x = np.arange(0,0.0030,0.0010)
            y = np.arange(0,0.0030,0.0010)
            plt.plot(x,y,color='cadetblue',linestyle='--',alpha=0.50)
            plt.scatter(y_pred,test_y,color='teal')

            plt.xlabel('Predicted CH4 Diffusivity (cm2/s)')
            plt.ylabel('Real CH4 Diffusivity (cm2/s)')
            plt.title('Plot of Real vs Predicted CH4 Diffusivity. Train=Good, Test=Good')
            text = f'R2: {r2:.4f} \n RMSE: {Decimal(rmse):.2E} \n MAE: {Decimal(mae):.2E} \n SRCC: {srcc:.4f}'
            plt.text(x=0.0009,y=0.00020,s=text)
            plt.savefig(r"C:\Users\Grace\Documents\Code\USRO 2025W\My Code\DS06_data\CH4_diffusion\Correlation_Plots\CH4_good_good.png")
            plt.close()
        
        elif dataset=='CH4_bad':
            train_x = good_x 
            test_x = bad_x 
            train_y = good_y 
            test_y = bad_y 

            r2,rmse,mae,srcc,y_pred = RFR_CH4(train_x.drop(columns=['MOF name']),test_x.drop(columns=['MOF name']),train_y,test_y)

            # Output Metrics
            with open(output_fp,'a') as results:
                results.write('\n CH4 Diffusivity Data. Train = Good, Test = Bad, Split = 80/20')
                results.write('\n Number of MOFs used in Training (Good): ' + str(len(train_x)))
                results.write('\n List of MOFs used in Training (Good): \n' + train_x['MOF name'].to_string())
                results.write('='*50)
                results.write('\n Number of MOFs used in Testing (Bad): ' + str(len(test_x)))
                results.write('\n List of MOFs used in Testing (Bad): \n' + test_x['MOF name'].to_string())
                results.write('='*50)
                results.write('\n Metrics of Evaluation: (Good/Bad)')
                results.write('\n R2 Score: ' + str(r2))
                results.write('\n RMSE: ' + str(rmse))
                results.write('\n MAE: ' + str(mae))
                results.write('\n SRCC: ' + str(srcc))
                results.write('\n' + '=+='*20)
            
            # Correlation Plot
            fig = plt.figure(figsize=(7,7),dpi=100)
            plt.rc('axes',axisbelow=True)
            plt.rcParams['font.size'] = 12
            plt.rcParams['font.family'] = 'sans-serif'
            plt.grid(color = 'lightgrey',linestyle='-')
            plt.ylim(ymin=0,ymax=0.0013)
            plt.xlim(xmin=0,xmax=0.0013)
            plt.yticks(fontsize=8)
            plt.xticks(fontsize=8)

            x = np.arange(0,0.0030,0.0010)
            y = np.arange(0,0.0030,0.0010)
            plt.plot(x,y,color='slateblue',linestyle='--',alpha=0.50)
            plt.scatter(y_pred,test_y,color='darkslateblue')

            plt.xlabel('Predicted CH4 Diffusivity (cm2/s)')
            plt.ylabel('Real CH4 Diffusivity (cm2/s)')
            plt.title('Plot of Real vs Predicted CH4 Diffusivity. Train=Good, Test=Bad')
            text = f'R2: {r2:.4f} \n RMSE: {Decimal(rmse):.2E} \n MAE: {Decimal(mae):.2E} \n SRCC: {srcc:.4f}'
            plt.text(x=0.0009,y=0.00020,s=text)
            plt.savefig(r"C:\Users\Grace\Documents\Code\USRO 2025W\My Code\DS06_data\CH4_diffusion\Correlation_Plots\CH4_good_bad.png")
            plt.close()

        elif dataset == 'CH4_mixed':
            train_x,test_x_good,train_y,test_y_good = train_test_split(good_x,good_y,train_size=0.80,test_size=0.20,random_state=42)
            bad_20 = bad.sample(n=len(test_x_good))
            test_x_bad = bad_20.drop(columns=['U_CH4 (mol/kg)','D_CH4 (cm2/s)'])
            test_y_bad = bad_20['D_CH4 (cm2/s)']
            test_x = pd.concat([test_x_good,test_x_bad])
            test_y = pd.concat([test_y_good,test_y_bad])

            r2,rmse,mae,srcc,y_pred = RFR_CH4(train_x.drop(columns=['MOF name']),test_x.drop(columns=['MOF name']),train_y,test_y)

            # Output Metrics
            with open(output_fp,'a') as results:
                results.write('\n CH4 Diffusivity Data. Train = Good, Test = Mixed, Split = 80/(20:20)')
                results.write('\n Number of MOFs used in Training (Good): ' + str(len(train_x)))
                results.write('\n List of MOFs used in Training (Good): \n' + train_x['MOF name'].to_string())
                results.write('='*50)
                results.write('\n Number of MOFs used in Testing (Mixed): ' + str(len(test_x)))
                results.write('\n List of MOFs used in Testing (Mixed): \n' + test_x['MOF name'].to_string())
                results.write('='*50)
                results.write('\n Metrics of Evaluation: (Good/Mixed)')
                results.write('\n R2 Score: ' + str(r2))
                results.write('\n RMSE: ' + str(rmse))
                results.write('\n MAE: ' + str(mae))
                results.write('\n SRCC: ' + str(srcc))
                results.write('\n' + '=+='*20)
            
            # Correlation Plot
            fig = plt.figure(figsize=(7,7),dpi=100)
            plt.rc('axes',axisbelow=True)
            plt.rcParams['font.size'] = 12
            plt.rcParams['font.family'] = 'sans-serif'
            plt.grid(color = 'lightgrey',linestyle='-')
            plt.ylim(ymin=0,ymax=0.0013)
            plt.xlim(xmin=0,xmax=0.0013)
            plt.yticks(fontsize=8)
            plt.xticks(fontsize=8)

            x = np.arange(0,0.0030,0.0010)
            y = np.arange(0,0.0030,0.0010)
            plt.plot(x,y,color='plum',linestyle='--',alpha=0.50)
            plt.scatter(y_pred,test_y,color='mediumvioletred')

            plt.xlabel('Predicted CH4 Diffusivity (cm2/s)')
            plt.ylabel('Real CH4 Diffusivity (cm2/s)')
            plt.title('Plot of Real vs Predicted CH4 Diffusivity. Train=Good, Test=Mixed')
            text = f'R2: {r2:.4f} \n RMSE: {Decimal(rmse):.2E} \n MAE: {Decimal(mae):.2E} \n SRCC: {srcc:.4f}'
            plt.text(x=0.0009,y=0.00020,s=text)
            plt.savefig(r"C:\Users\Grace\Documents\Code\USRO 2025W\My Code\DS06_data\CH4_diffusion\Correlation_Plots\CH4_good_mixed.png")
            plt.close()

def train_mixed(datasets):
    for dataset in datasets:
        if dataset == 'CH4_mixed':
            mixed = pd.concat([good,bad])
            mixed_x = mixed.drop(columns=['U_CH4 (mol/kg)','D_CH4 (cm2/s)'])
            mixed_y = mixed['D_CH4 (cm2/s)']
            train_x,test_x,train_y,test_y = train_test_split(mixed_x,mixed_y,train_size=0.80,test_size=0.20,random_state=42)

            r2,rmse,mae,srcc,y_pred = RFR_CH4(train_x.drop(columns=['MOF name']),test_x.drop(columns=['MOF name']),train_y,test_y)

            # Output Metrics
            with open(output_fp,'a') as results:
                results.write('\n CH4 Diffusivity Data. Train = Mixed, Test = Mixed, Split = 80/20')
                results.write('\n Number of MOFs used in Training (Mixed): ' + str(len(train_x)))
                results.write('\n List of MOFs used in Training (Mixed): \n' + train_x['MOF name'].to_string())
                results.write('='*50)
                results.write('\n Number of MOFs used in Testing (Mixed): ' + str(len(test_x)))
                results.write('\n List of MOFs used in Testing (Mixed): \n' + test_x['MOF name'].to_string())
                results.write('='*50)
                results.write('\n Metrics of Evaluation: (Mixed/Mixed)')
                results.write('\n R2 Score: ' + str(r2))
                results.write('\n RMSE: ' + str(rmse))
                results.write('\n MAE: ' + str(mae))
                results.write('\n SRCC: ' + str(srcc))
                results.write('\n' + '=+='*20)
            
            # Correlation Plot
            fig = plt.figure(figsize=(7,7),dpi=100)
            plt.rc('axes',axisbelow=True)
            plt.rcParams['font.size'] = 12
            plt.rcParams['font.family'] = 'sans-serif'
            plt.grid(color = 'lightgrey',linestyle='-')
            plt.ylim(ymin=0,ymax=0.0013)
            plt.xlim(xmin=0,xmax=0.0013)
            plt.yticks(fontsize=8)
            plt.xticks(fontsize=8)

            x = np.arange(0,0.0030,0.0010)
            y = np.arange(0,0.0030,0.0010)
            plt.plot(x,y,color='royalblue',linestyle='--',alpha=0.50)
            plt.scatter(y_pred,test_y,color='navy')

            plt.xlabel('Predicted CH4 Diffusivity (cm2/s)')
            plt.ylabel('Real CH4 Diffusivity (cm2/s)')
            plt.title('Plot of Real vs Predicted CH4 Diffusivity. Train=Mixed, Test=Mixed')
            text = f'R2: {r2:.4f} \n RMSE: {Decimal(rmse):.2E} \n MAE: {Decimal(mae):.2E} \n SRCC: {srcc:.4f}'
            plt.text(x=0.0009,y=0.00020,s=text)
            plt.savefig(r"C:\Users\Grace\Documents\Code\USRO 2025W\My Code\DS06_data\CH4_diffusion\Correlation_Plots\CH4_mixed_mixed.png")
            plt.close()
        
        elif dataset == 'CH4_good':
            train_good_x,test_good_x,train_good_y,test_good_y = train_test_split(good_x,good_y,train_size=0.80,test_size=0.20)
            bad_80 = bad.sample(frac=0.80)
            train_bad_x = bad_80.drop(columns=['U_CH4 (mol/kg)','D_CH4 (cm2/s)'])
            train_bad_y = bad_80['D_CH4 (cm2/s)']
            train_x = pd.concat([train_good_x,train_bad_x])
            train_y = pd.concat([train_good_y,train_bad_y])

            r2,rmse,mae,srcc,y_pred = RFR_CH4(
                train_x.drop(columns=['MOF name']),
                test_good_x.drop(columns=['MOF name']),
                train_y,test_good_y
            )

            # Output Metrics
            with open(output_fp,'a') as results:
                results.write('\n CH4 Diffusivity Data. Train = Mixed, Test = Good, Split = (80+80)/20')
                results.write('\n Number of MOFs used in Training (Mixed): ' + str(len(train_x)))
                results.write('\n List of MOFs used in Training (Mixed): \n' + train_x['MOF name'].to_string())
                results.write('='*50)
                results.write('\n Number of MOFs used in Testing (Good): ' + str(len(test_good_x)))
                results.write('\n List of MOFs used in Testing (Good): \n' + test_good_x['MOF name'].to_string())
                results.write('='*50)
                results.write('\n Metrics of Evaluation: (Mixed/Good)')
                results.write('\n R2 Score: ' + str(r2))
                results.write('\n RMSE: ' + str(rmse))
                results.write('\n MAE: ' + str(mae))
                results.write('\n SRCC: ' + str(srcc))
                results.write('\n' + '=+='*20)
            
            # Correlation Plot
            fig = plt.figure(figsize=(7,7),dpi=100)
            plt.rc('axes',axisbelow=True)
            plt.rcParams['font.size'] = 12
            plt.rcParams['font.family'] = 'sans-serif'
            plt.grid(color = 'lightgrey',linestyle='-')
            plt.ylim(ymin=0,ymax=0.0013)
            plt.xlim(xmin=0,xmax=0.0013)
            plt.yticks(fontsize=8)
            plt.xticks(fontsize=8)

            x = np.arange(0,0.0030,0.0010)
            y = np.arange(0,0.0030,0.0010)
            plt.plot(x,y,color='palevioletred',linestyle='--',alpha=0.50)
            plt.scatter(y_pred,test_good_y,color='mediumvioletred')

            plt.xlabel('Predicted CH4 Diffusivity (cm2/s)')
            plt.ylabel('Real CH4 Diffusivity (cm2/s)')
            plt.title('Plot of Real vs Predicted CH4 Diffusivity. Train=Mixed, Test=Good')
            text = f'R2: {r2:.4f} \n RMSE: {Decimal(rmse):.2E} \n MAE: {Decimal(mae):.2E} \n SRCC: {srcc:.4f}'
            plt.text(x=0.0009,y=0.00020,s=text)
            plt.savefig(r"C:\Users\Grace\Documents\Code\USRO 2025W\My Code\DS06_data\CH4_diffusion\Correlation_Plots\CH4_mixed_good.png")
            plt.close()
        
        elif dataset == 'CH4_bad':
            train_bad_x,test_bad_x,train_bad_y,test_bad_y = train_test_split(bad_x,bad_y,train_size=0.80,test_size=0.20,random_state=42)
            good_80 = good.sample(frac=0.80)
            train_x = pd.concat([train_bad_x,good_80.drop(columns=['U_CH4 (mol/kg)','D_CH4 (cm2/s)'])])
            test_x = test_bad_x 
            train_y = pd.concat([train_bad_y,good_80['D_CH4 (cm2/s)']])
            test_y = test_bad_y

            r2,rmse,mae,srcc,y_pred = RFR_CH4(train_x.drop(columns=['MOF name']),test_x.drop(columns=['MOF name']),train_y,test_y)

            # Output Metrics
            with open(output_fp,'a') as results:
                results.write('\n CH4 Diffusivity Data. Train = Mixed, Test = Bad, Split = (80+80)/20')
                results.write('\n Number of MOFs used in Training (Mixed): ' + str(len(train_x)))
                results.write('\n List of MOFs used in Training (Mixed): \n' + train_x['MOF name'].to_string())
                results.write('='*50)
                results.write('\n Number of MOFs used in Testing (Bad): ' + str(len(test_x)))
                results.write('\n List of MOFs used in Testing (Bad): \n' + test_x['MOF name'].to_string())
                results.write('='*50)
                results.write('\n Metrics of Evaluation: (Mixed/Bad)')
                results.write('\n R2 Score: ' + str(r2))
                results.write('\n RMSE: ' + str(rmse))
                results.write('\n MAE: ' + str(mae))
                results.write('\n SRCC: ' + str(srcc))
                results.write('\n' + '=+='*20)
            
            # Correlation Plot
            fig = plt.figure(figsize=(7,7),dpi=100)
            plt.rc('axes',axisbelow=True)
            plt.rcParams['font.size'] = 12
            plt.rcParams['font.family'] = 'sans-serif'
            plt.grid(color = 'lightgrey',linestyle='-')
            plt.ylim(ymin=0,ymax=0.0013)
            plt.xlim(xmin=0,xmax=0.0013)
            plt.yticks(fontsize=8)
            plt.xticks(fontsize=8)

            x = np.arange(0,0.0030,0.0010)
            y = np.arange(0,0.0030,0.0010)
            plt.plot(x,y,color='darkkhaki',linestyle='--',alpha=0.50)
            plt.scatter(y_pred,test_y,color='darkgoldenrod')

            plt.xlabel('Predicted CH4 Diffusivity (cm2/s)')
            plt.ylabel('Real CH4 Diffusivity (cm2/s)')
            plt.title('Plot of Real vs Predicted CH4 Diffusivity. Train=Mixed, Test=Bad')
            text = f'R2: {r2:.4f} \n RMSE: {Decimal(rmse):.2E} \n MAE: {Decimal(mae):.2E} \n SRCC: {srcc:.4f}'
            plt.text(x=0.0009,y=0.00020,s=text)
            plt.savefig(r"C:\Users\Grace\Documents\Code\USRO 2025W\My Code\DS06_data\CH4_diffusion\Correlation_Plots\CH4_mixed_bad.png")
            plt.close()
