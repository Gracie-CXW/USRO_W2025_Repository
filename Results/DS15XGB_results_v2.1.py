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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import math
from decimal import Decimal

#importing raw data, and splitting.
'''
args = argv      # input data filepath here, should be .csv format
fp = args[1]
'''

# Importing Data
fp = r"C:\Users\Grace\Documents\Code\USRO 2025W\Raw_Data\DS15\jp3c02452_si_003.csv"
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


Dataset = pd.read_csv(fp)

good = Dataset[Dataset['name'].isin(good_names)]
bad = Dataset[Dataset['name'].isin(bad_names)]

good_x = good.drop(columns=['S','OMS type'])
good_y = good['S']
bad_x = bad.drop(columns=['S','OMS type'])
bad_y = bad['S']

# Data preprocessing - paper did not specify whether the features were preprocessed
# prior to training, nor the type of preprocessing for each feature. 
# I am using standard scaler

def XGB(train_x,test_x,train_y,test_y):
    import xgboost
    from xgboost import XGBRegressor

    train_x = train_x 
    test_x = test_x 
    train_y = train_y 
    test_y = test_y 

    # Scaling
    scaler = StandardScaler().fit(train_x)
    train_x = scaler.transform(train_x)
    test_x = scaler.transform(test_x)

    # Model
    xgb = XGBRegressor(
        learing_rate = 0.05,
        max_depth = 10,
        n_estimators = 1500,
        random_state = 42
    )
    xgb.fit(train_x,train_y)

    pred_y = xgb.predict(test_x)
    r2 = r2_score(pred_y,test_y)
    rmse = root_mean_squared_error(pred_y,test_y)
    mape = mean_absolute_percentage_error(pred_y,test_y)

    return pred_y,r2,rmse,mape

tests = ['bad','good','mixed']

def train_bad(datasets,output):
    output=output
    bad_bad = []
    bad_good = []
    bad_mixed = []

    for dataset in datasets:
        if dataset == 'bad':
            train_x,test_x,train_y,test_y = train_test_split(bad_x,bad_y,train_size=0.80,test_size=0.20,random_state=42)

            pred_y,r2,rmse,mape = XGB(
                train_x.drop(columns=['name']),
                test_x.drop(columns=['name']),
                train_y,test_y)

            with open(output,'a') as results:
                results.write('=+'*25)
                results.write('\n Evaluation Metrics for CO2-to-CO Selectivity, S, for DS15 paper. (Bad/Bad) Split = 80/20')
                results.write('\n Number of MOFs used in Training: (bad/bad)' + str(len(train_x)))
                results.write('\n List of MOFs used in Training: (bad/bad)' + train_x['name'].to_string())
                results.write('\n' + '='*50)
                results.write('\n Number of MOFs used in Testing: (bad/bad)' + str(len(test_x)))
                results.write('\n List of MOFs used in Training: (bad/bad)' + test_x['name'].to_string())
                results.write('\n' + '='*50)
                results.write('\n Metrics of Evaluation (bad/bad):')
                results.write('\n R2 Score: ' + str(r2))
                results.write('\n RMSE: ' + str(rmse))
                results.write('\n MAPE: ' + str(mape))
            
            bad_bad.append(r2)
            bad_bad.append(rmse)
            bad_bad.append(mape)

            # Correlation Plot
            fig = plt.figure(figsize=(8,8),dpi=100)
            plt.rc('axes',axisbelow=True)
            plt.rcParams['font.size'] = 12
            plt.rcParams['font.family'] = 'DejaVu Sans'
            plt.grid(color = 'lightgrey',linestyle='-')

            max_y = 1
            if max(test_y)>=max(pred_y):
                max_y = max(test_y)
            else: 
                max_y = max(pred_y)
            
            plt.ylim(ymin=0,ymax=max_y+50)
            plt.xlim(xmin=0,xmax=max_y+50)
            plt.yticks(fontsize=8)
            plt.xticks(fontsize=8)

            x = np.arange(0,10000,1000)
            y = np.arange(0,10000,1000)
            plt.plot(x,y,color='salmon',linestyle='--',alpha=0.50)
            plt.scatter(pred_y,test_y,color='orangered',alpha=0.80)

            plt.xlabel('Predicted CO2-to-CO Selectivity, S')
            plt.ylabel('Real CO2-to-CO Selectivity, S')
            plt.title('Plot of Real vs Predicted CO2-to-CO Selectivty. Train=Bad, Test=Bad')
            text = f'R2: {r2:.4f} \n RMSE: {Decimal(rmse):.2E} \n MAPE: {mape:.3f} \n Train Size: {len(train_x)} \n Test Size: {len(test_x)}'
            plt.text(x=max_y*0.70,y=max_y*0.10,s=text)
            plt.savefig(r"C:\Users\Grace\Documents\Code\USRO 2025W\My Code\DS15_data\Correlation_Plots\DS15_bad_bad.png")
            plt.close()
        
        elif dataset == 'good':
            train_x = bad_x 
            test_x = good_x
            train_y = bad_y 
            test_y = good_y 

            pred_y,r2,rmse,mape = XGB(
                train_x.drop(columns=['name']),
                test_x.drop(columns=['name']),
                train_y,test_y)

            with open(output,'a') as results:
                results.write('=+'*25)
                results.write('\n Evaluation Metrics for CO2-to-CO Selectivity, S, for DS15 paper. (Bad/Good) No Split')
                results.write('\n Number of MOFs used in Training: (bad/good)' + str(len(train_x)))
                results.write('\n List of MOFs used in Training: (bad/good)' + train_x['name'].to_string())
                results.write('\n' + '='*50)
                results.write('\n Number of MOFs used in Testing: (bad/good)' + str(len(test_x)))
                results.write('\n List of MOFs used in Training: (bad/good)' + test_x['name'].to_string())
                results.write('\n' + '='*50)
                results.write('\n Metrics of Evaluation (bad/good):')
                results.write('\n R2 Score: ' + str(r2))
                results.write('\n RMSE: ' + str(rmse))
                results.write('\n MAPE: ' + str(mape))

            bad_good.append(r2)
            bad_good.append(rmse)
            bad_good.append(mape)

            # Correlation Plot
            fig = plt.figure(figsize=(8,8),dpi=100)
            plt.rc('axes',axisbelow=True)
            plt.rcParams['font.size'] = 12
            plt.rcParams['font.family'] = 'DejaVu Sans'
            plt.grid(color = 'lightgrey',linestyle='-')

            max_y = 1
            if max(test_y)>=max(pred_y):
                max_y = max(test_y)
            else: 
                max_y = max(pred_y)

            plt.ylim(ymin=0,ymax=max_y+50)
            plt.xlim(xmin=0,xmax=max_y+50)
            plt.yticks(fontsize=8)
            plt.xticks(fontsize=8)

            x = np.arange(0,10000,1000)
            y = np.arange(0,10000,1000)
            plt.plot(x,y,color='yellowgreen',linestyle='--',alpha=0.50)
            plt.scatter(pred_y,test_y,color='forestgreen',alpha=0.80)

            plt.xlabel('Predicted CO2-to-CO Selectivity, S')
            plt.ylabel('Real CO2-to-CO Selectivity, S')
            plt.title('Plot of Real vs Predicted CO2-to-CO Selectivty. Train=Bad, Test=Good')
            text = f'R2: {r2:.4f} \n RMSE: {Decimal(rmse):.2E} \n MAPE: {mape:.3f} \n Train Size: {len(train_x)} \n Test Size: {len(test_x)}'
            plt.text(x=max_y*0.70,y=max_y*0.10,s=text)
            plt.savefig(r"C:\Users\Grace\Documents\Code\USRO 2025W\My Code\DS15_data\Correlation_Plots\DS15_bad_good.png")
            plt.close()
        
        elif dataset == 'mixed':
            train_x,test_x_bad,train_y,test_y_bad = train_test_split(bad_x,bad_y,train_size=0.80,test_size=0.20)
            test_good = good.sample(n=len(test_x_bad))
            test_x = pd.concat([test_x_bad,test_good.drop(columns=['S','OMS type'])])
            test_y = pd.concat([test_y_bad,test_good['S']])

            pred_y,r2,rmse,mape = XGB(
                train_x.drop(columns=['name']),
                test_x.drop(columns=['name']),
                train_y,test_y)

            with open(output,'a') as results:
                results.write('=+'*25)
                results.write('\n Evaluation Metrics for CO2-to-CO Selectivity, S, for DS15 paper. (Bad/Mixed) Split: 80/(20:20)')
                results.write('\n Number of MOFs used in Training: (bad/mixed)' + str(len(train_x)))
                results.write('\n List of MOFs used in Training: (bad/mixed)' + train_x['name'].to_string())
                results.write('\n' + '='*50)
                results.write('\n Number of MOFs used in Testing: (bad/mixed)' + str(len(test_x)))
                results.write('\n List of MOFs used in Training: (bad/mixed)' + test_x['name'].to_string())
                results.write('\n' + '='*50)
                results.write('\n Metrics of Evaluation (bad/mixed):')
                results.write('\n R2 Score: ' + str(r2))
                results.write('\n RMSE: ' + str(rmse))
                results.write('\n MAPE: ' + str(mape))
            
            bad_mixed.append(r2)
            bad_mixed.append(rmse)
            bad_mixed.append(mape)

            # Correlation Plot
            fig = plt.figure(figsize=(8,8),dpi=100)
            plt.rc('axes',axisbelow=True)
            plt.rcParams['font.size'] = 12
            plt.rcParams['font.family'] = 'DejaVu Sans'
            plt.grid(color = 'lightgrey',linestyle='-')

            max_y = 1
            if max(test_y)>=max(pred_y):
                max_y = max(test_y)
            else: 
                max_y = max(pred_y)

            plt.ylim(ymin=0,ymax=max_y+50)
            plt.xlim(xmin=0,xmax=max_y+50)
            plt.yticks(fontsize=8)
            plt.xticks(fontsize=8)

            x = np.arange(0,10000,1000)
            y = np.arange(0,10000,1000)
            plt.plot(x,y,color='mediumaquamarine',linestyle='--',alpha=0.50)
            plt.scatter(pred_y,test_y,color='lightseagreen',alpha=0.80)

            plt.xlabel('Predicted CO2-to-CO Selectivity, S')
            plt.ylabel('Real CO2-to-CO Selectivity, S')
            plt.title('Plot of Real vs Predicted CO2-to-CO Selectivty. Train=Bad, Test=Mixed')
            text = f'R2: {r2:.4f} \n RMSE: {Decimal(rmse):.2E} \n MAPE: {mape:.3f} \n Train Size: {len(train_x)} \n Test Size: {len(test_x)}'
            plt.text(x=max_y*0.70, y=max_y*0.10, s=text)
            plt.savefig(r"C:\Users\Grace\Documents\Code\USRO 2025W\My Code\DS15_data\Correlation_Plots\DS15_bad_mixed.png")
            plt.close()
        
    with open(output,'a') as finals:
        finals.write('\n' + '#'*50)
        finals.write('\n Metrics of Evaluation. (BAD/BAD)')
        finals.write('\n R2 (bad/bad): ' + str(bad_bad[0]))
        finals.write('\n RMSE (bad/bad): ' + str(bad_bad[1]))
        finals.write('\n MAPE (bad/bad): ' + str(bad_bad[2]))

        finals.write('\n Metrics of Evaluation. (BAD/GOOD)')
        finals.write('\n R2 (bad/good): ' + str(bad_good[0]))
        finals.write('\n RMSE (bad/good): ' + str(bad_good[1]))
        finals.write('\n MAPE (bad/good): ' + str(bad_good[2]))

        finals.write('\n Metrics of Evaluation. (BAD/MIXED)')
        finals.write('\n R2 (bad/mixed): ' + str(bad_mixed[0]))
        finals.write('\n RMSE (bad/mixed): ' + str(bad_mixed[1]))
        finals.write('\n MAPE (bad/mixed): ' + str(bad_mixed[2]))


def train_good(datasets,output):
    output = output
    good_good = []
    good_bad = []
    good_mixed = []

    for dataset in datasets:
        if dataset == 'good':
            train_x,test_x,train_y,test_y = train_test_split(good_x,good_y,train_size=0.80,test_size=0.20,random_state=42)

            pred_y,r2,rmse,mape = XGB(
                train_x.drop(columns=['name']),
                test_x.drop(columns=['name']),
                train_y,test_y)

            with open(output,'a') as results:
                results.write('=+'*25)
                results.write('\n Evaluation Metrics for CO2-to-CO Selectivity, S, for DS15 paper. (Good/Good) Split = 80/20')
                results.write('\n Number of MOFs used in Training: (Good/Good)' + str(len(train_x)))
                results.write('\n List of MOFs used in Training: (Good/Good)' + train_x['name'].to_string())
                results.write('\n' + '='*50)
                results.write('\n Number of MOFs used in Testing: (Good/Good)' + str(len(test_x)))
                results.write('\n List of MOFs used in Training: (Good/Good)' + test_x['name'].to_string())
                results.write('\n' + '='*50)
                results.write('\n Metrics of Evaluation (Good/Good):')
                results.write('\n R2 Score: ' + str(r2))
                results.write('\n RMSE: ' + str(rmse))
                results.write('\n MAPE: ' + str(mape))
            
            good_good.append(r2)
            good_good.append(rmse)
            good_good.append(mape)

            # Correlation Plot
            fig = plt.figure(figsize=(8,8),dpi=100)
            plt.rc('axes',axisbelow=True)
            plt.rcParams['font.size'] = 12
            plt.rcParams['font.family'] = 'DejaVu Sans'
            plt.grid(color = 'lightgrey',linestyle='-')

            max_y = 1
            if max(test_y)>=max(pred_y):
                max_y = max(test_y)
            else: 
                max_y = max(pred_y)
            
            plt.ylim(ymin=0,ymax=max_y+50)
            plt.xlim(xmin=0,xmax=max_y+50)
            plt.yticks(fontsize=8)
            plt.xticks(fontsize=8)

            x = np.arange(0,10000,1000)
            y = np.arange(0,10000,1000)
            plt.plot(x,y,color='lightseagreen',linestyle='--',alpha=0.50)
            plt.scatter(pred_y,test_y,color='darkcyan',alpha=0.80)

            plt.xlabel('Predicted CO2-to-CO Selectivity, S')
            plt.ylabel('Real CO2-to-CO Selectivity, S')
            plt.title('Plot of Real vs Predicted CO2-to-CO Selectivty. Train=Good, Test=Good')
            text = f'R2: {r2:.4f} \n RMSE: {rmse:.4f} \n MAPE: {mape:.3f} \n Train Size: {len(train_x)} \n Test Size: {len(test_x)}'
            plt.text(x=max_y*0.70,y=max_y*0.10,s=text)
            plt.savefig(r"C:\Users\Grace\Documents\Code\USRO 2025W\My Code\DS15_data\Correlation_Plots\DS15_good_good.png")
            plt.close()
        
        elif dataset == 'bad':
            train_x = good_x 
            test_x = bad_x 
            train_y = good_y 
            test_y = bad_y 

            pred_y,r2,rmse,mape = XGB(
                train_x.drop(columns=['name']),
                test_x.drop(columns=['name']),
                train_y,test_y)

            with open(output,'a') as results:
                results.write('=+'*25)
                results.write('\n Evaluation Metrics for CO2-to-CO Selectivity, S, for DS15 paper. (Good/Bad) No Split')
                results.write('\n Number of MOFs used in Training: (Good/Bad)' + str(len(train_x)))
                results.write('\n List of MOFs used in Training: (Good/Bad)' + train_x['name'].to_string())
                results.write('\n' + '='*50)
                results.write('\n Number of MOFs used in Testing: (Good/Bad)' + str(len(test_x)))
                results.write('\n List of MOFs used in Training: (Good/Bad)' + test_x['name'].to_string())
                results.write('\n' + '='*50)
                results.write('\n Metrics of Evaluation (Good/Bad):')
                results.write('\n R2 Score: ' + str(r2))
                results.write('\n RMSE: ' + str(rmse))
                results.write('\n MAPE: ' + str(mape))
            
            good_bad.append(r2)
            good_bad.append(rmse)
            good_bad.append(mape)

            # Correlation Plot
            fig = plt.figure(figsize=(8,8),dpi=100)
            plt.rc('axes',axisbelow=True)
            plt.rcParams['font.size'] = 12
            plt.rcParams['font.family'] = 'DejaVu Sans'
            plt.grid(color = 'lightgrey',linestyle='-')

            max_y = 1
            if max(test_y)>=max(pred_y):
                max_y = max(test_y)
            else: 
                max_y = max(pred_y)
            
            plt.ylim(ymin=0,ymax=max_y+50)
            plt.xlim(xmin=0,xmax=max_y+50)
            plt.yticks(fontsize=8)
            plt.xticks(fontsize=8)

            x = np.arange(0,10000,1000)
            y = np.arange(0,10000,1000)
            plt.plot(x,y,color='lightseagreen',linestyle='--',alpha=0.50)
            plt.scatter(pred_y,test_y,color='darkcyan',alpha=0.80)

            plt.xlabel('Predicted CO2-to-CO Selectivity, S')
            plt.ylabel('Real CO2-to-CO Selectivity, S')
            plt.title('Plot of Real vs Predicted CO2-to-CO Selectivty. Train=Good, Test=Bad')
            text = f'R2: {r2:.4f} \n RMSE: {rmse:.4f} \n MAPE: {mape:.3f} \n Train Size: {len(train_x)} \n Test Size: {len(test_x)}'
            plt.text(x=max_y*0.70,y=max_y*0.10,s=text)
            plt.savefig(r"C:\Users\Grace\Documents\Code\USRO 2025W\My Code\DS15_data\Correlation_Plots\DS15_good_bad.png")
            plt.close()
        
        elif dataset == 'mixed':
            train_x,test_x_good,train_y,test_y_good = train_test_split(good_x,good_y,train_size=0.80,test_size=0.20,random_state=42)
            badset = bad.sample(n=len(test_x_good))
            test_x = pd.concat([test_x_good,badset.drop(columns=['S','OMS type'])])
            test_y = pd.concat([test_y_good,badset['S']])

            pred_y,r2,rmse,mape = XGB(
                train_x.drop(columns=['name']),
                test_x.drop(columns=['name']),
                train_y,test_y)

            with open(output,'a') as results:
                results.write('=+'*25)
                results.write('\n Evaluation Metrics for CO2-to-CO Selectivity, S, for DS15 paper. (Good/Mixed) Split = 80/(20:20)')
                results.write('\n Number of MOFs used in Training: (Good/Mixed)' + str(len(train_x)))
                results.write('\n List of MOFs used in Training: (Good/Mixed)' + train_x['name'].to_string())
                results.write('\n' + '='*50)
                results.write('\n Number of MOFs used in Testing: (Good/Mixed)' + str(len(test_x)))
                results.write('\n List of MOFs used in Training: (Good/Mixed)' + test_x['name'].to_string())
                results.write('\n' + '='*50)
                results.write('\n Metrics of Evaluation (Good/Mixed):')
                results.write('\n R2 Score: ' + str(r2))
                results.write('\n RMSE: ' + str(rmse))
                results.write('\n MAPE: ' + str(mape))
            
            good_mixed.append(r2)
            good_mixed.append(rmse)
            good_mixed.append(mape)

            # Correlation Plot
            fig = plt.figure(figsize=(8,8),dpi=100)
            plt.rc('axes',axisbelow=True)
            plt.rcParams['font.size'] = 12
            plt.rcParams['font.family'] = 'DejaVu Sans'
            plt.grid(color = 'lightgrey',linestyle='-')

            max_y = 1
            if max(test_y)>=max(pred_y):
                max_y = max(test_y)
            else: 
                max_y = max(pred_y)
            
            plt.ylim(ymin=0,ymax=max_y+50)
            plt.xlim(xmin=0,xmax=max_y+50)
            plt.yticks(fontsize=8)
            plt.xticks(fontsize=8)

            x = np.arange(0,10000,1000)
            y = np.arange(0,10000,1000)
            plt.plot(x,y,color='lightseagreen',linestyle='--',alpha=0.50)
            plt.scatter(pred_y,test_y,color='darkcyan',alpha=0.80)

            plt.xlabel('Predicted CO2-to-CO Selectivity, S')
            plt.ylabel('Real CO2-to-CO Selectivity, S')
            plt.title('Plot of Real vs Predicted CO2-to-CO Selectivty. Train=Good, Test=Mixed')
            text = f'R2: {r2:.4f} \n RMSE: {rmse:.4f} \n MAPE: {mape:.3f} \n Train Size: {len(train_x)} \n Test Size: {len(test_x)}'
            plt.text(x=max_y*0.70,y=max_y*0.10,s=text)
            plt.savefig(r"C:\Users\Grace\Documents\Code\USRO 2025W\My Code\DS15_data\Correlation_Plots\DS15_good_mixed.png")
            plt.close()
        
    with open(output,'a') as finals:
        finals.write('\n' + '#'*50)
        finals.write('\n Metrics of Evaluation. (GOOD/GOOD)')
        finals.write('\n R2 (good/good): ' + str(good_good[0]))
        finals.write('\n RMSE (good/good): ' + str(good_good[1]))
        finals.write('\n MAPE (good/good): ' + str(good_good[2]))

        finals.write('\n Metrics of Evaluation. (GOOD/BAD)')
        finals.write('\n R2 (good/bad): ' + str(good_bad[0]))
        finals.write('\n RMSE (good/bad): ' + str(good_bad[1]))
        finals.write('\n MAPE (good/bad): ' + str(good_bad[2]))

        finals.write('\n Metrics of Evaluation. (GOOD/MIXED)')
        finals.write('\n R2 (good/mixed): ' + str(good_mixed[0]))
        finals.write('\n RMSE (good/mixed): ' + str(good_mixed[1]))
        finals.write('\n MAPE (good/mixed): ' + str(good_mixed[2]))

def train_mixed(datasets,output):
    output=output
    mixed_good = []
    mixed_bad = []
    mixed_mixed = []

    for dataset in datasets:
        if dataset == 'mixed':
            mixed = pd.concat([good,bad])
            mixed_x = mixed.drop(columns = ['S','OMS type'])
            mixed_y = mixed['S']
            train_x,test_x,train_y,test_y = train_test_split(mixed_x,mixed_y,train_size=0.80,test_size=0.20,random_state=42)

            pred_y,r2,rmse,mape = XGB(
                train_x.drop(columns=['name']),
                test_x.drop(columns=['name']),
                train_y,test_y)

            with open(output,'a') as results:
                results.write('=+'*25)
                results.write('\n Evaluation Metrics for CO2-to-CO Selectivity, S, for DS15 paper. (Mixed/Mixed) Split = 80/20')
                results.write('\n Number of MOFs used in Training: (Mixed/Mixed)' + str(len(train_x)))
                results.write('\n List of MOFs used in Training: (Mixed/Mixed)' + train_x['name'].to_string())
                results.write('\n' + '='*50)
                results.write('\n Number of MOFs used in Testing: (Mixed/Mixed)' + str(len(test_x)))
                results.write('\n List of MOFs used in Training: (Mixed/Mixed)' + test_x['name'].to_string())
                results.write('\n' + '='*50)
                results.write('\n Metrics of Evaluation (Mixed/Mixed):')
                results.write('\n R2 Score: ' + str(r2))
                results.write('\n RMSE: ' + str(rmse))
                results.write('\n MAPE: ' + str(mape))
            
            mixed_mixed.append(r2)
            mixed_mixed.append(rmse)
            mixed_mixed.append(mape)

            # Correlation Plot
            fig = plt.figure(figsize=(8,8),dpi=100)
            plt.rc('axes',axisbelow=True)
            plt.rcParams['font.size'] = 12
            plt.rcParams['font.family'] = 'DejaVu Sans'
            plt.grid(color = 'lightgrey',linestyle='-')

            max_y = 1
            if max(test_y)>=max(pred_y):
                max_y = max(test_y)
            else: 
                max_y = max(pred_y)
            
            plt.ylim(ymin=0,ymax=max_y+50)
            plt.xlim(xmin=0,xmax=max_y+50)
            plt.yticks(fontsize=8)
            plt.xticks(fontsize=8)

            x = np.arange(0,10000,1000)
            y = np.arange(0,10000,1000)
            plt.plot(x,y,color='violet',linestyle='--',alpha=0.50)
            plt.scatter(pred_y,test_y,color='purple',alpha=0.80)

            plt.xlabel('Predicted CO2-to-CO Selectivity, S')
            plt.ylabel('Real CO2-to-CO Selectivity, S')
            plt.title('Plot of Real vs Predicted CO2-to-CO Selectivty. Train=Mixed, Test=Mixed')
            text = f'R2: {r2:.4f} \n RMSE: {rmse:.4f} \n MAPE: {mape:.3f} \n Train Size: {len(train_x)} \n Test Size: {len(test_x)}'
            plt.text(x=max_y*0.70,y=max_y*0.10,s=text)
            plt.savefig(r"C:\Users\Grace\Documents\Code\USRO 2025W\My Code\DS15_data\Correlation_Plots\DS15_mixed_mixed.png")
            plt.close()
        
        elif dataset == 'good':
            train_x_good,test_x_good,train_y_good,test_y_good = train_test_split(good_x,good_y,train_size=0.80,test_size=0.20,random_state=42)
            train_x_bad,test_x_bad,train_y_bad,test_y_bad = train_test_split(bad_x,bad_y,train_size=0.80,test_size=0.20,random_state=42)
            train_x = pd.concat([train_x_good,train_x_bad])
            test_x = test_x_good
            train_y = pd.concat([train_y_good,train_y_bad])
            test_y = test_y_good

            pred_y,r2,rmse,mape = XGB(
                train_x.drop(columns=['name']),
                test_x.drop(columns=['name']),
                train_y,test_y)

            with open(output,'a') as results:
                results.write('=+'*25)
                results.write('\n Evaluation Metrics for CO2-to-CO Selectivity, S, for DS15 paper. (Mixed/Good) Split = (80+80)/20')
                results.write('\n Number of MOFs used in Training: (Mixed/Good)' + str(len(train_x)))
                results.write('\n List of MOFs used in Training: (Mixed/Good)' + train_x['name'].to_string())
                results.write('\n' + '='*50)
                results.write('\n Number of MOFs used in Testing: (Mixed/Good)' + str(len(test_x)))
                results.write('\n List of MOFs used in Training: (Mixed/Good)' + test_x['name'].to_string())
                results.write('\n' + '='*50)
                results.write('\n Metrics of Evaluation (Mixed/Good):')
                results.write('\n R2 Score: ' + str(r2))
                results.write('\n RMSE: ' + str(rmse))
                results.write('\n MAPE: ' + str(mape))
            
            mixed_good.append(r2)
            mixed_good.append(rmse)
            mixed_good.append(mape)

            # Correlation Plot
            fig = plt.figure(figsize=(8,8),dpi=100)
            plt.rc('axes',axisbelow=True)
            plt.rcParams['font.size'] = 12
            plt.rcParams['font.family'] = 'DejaVu Sans'
            plt.grid(color = 'lightgrey',linestyle='-')

            max_y = 1
            if max(test_y)>=max(pred_y):
                max_y = max(test_y)
            else: 
                max_y = max(pred_y)
            
            plt.ylim(ymin=0,ymax=max_y+50)
            plt.xlim(xmin=0,xmax=max_y+50)
            plt.yticks(fontsize=8)
            plt.xticks(fontsize=8)

            x = np.arange(0,10000,1000)
            y = np.arange(0,10000,1000)
            plt.plot(x,y,color='violet',linestyle='--',alpha=0.50)
            plt.scatter(pred_y,test_y,color='purple',alpha=0.80)

            plt.xlabel('Predicted CO2-to-CO Selectivity, S')
            plt.ylabel('Real CO2-to-CO Selectivity, S')
            plt.title('Plot of Real vs Predicted CO2-to-CO Selectivty. Train=Mixed, Test=Good')
            text = f'R2: {r2:.4f} \n RMSE: {rmse:.4f} \n MAPE: {mape:.3f} \n Train Size: {len(train_x)} \n Test Size: {len(test_x)}'
            plt.text(x=max_y*0.70,y=max_y*0.10,s=text)
            plt.savefig(r"C:\Users\Grace\Documents\Code\USRO 2025W\My Code\DS15_data\Correlation_Plots\DS15_mixed_good.png")
            plt.close()

        elif dataset == 'bad':
            train_x_bad,test_x_bad,train_y_bad,test_y_bad = train_test_split(bad_x,bad_y,train_size=0.80,test_size=0.20,random_state=42)
            train_x_good,test_x_good,train_y_good,test_y_good = train_test_split(good_x,good_y,train_size=0.80,test_size=0.20,random_state=42)
            train_x = pd.concat([train_x_good,train_x_bad])
            test_x = test_x_bad 
            train_y = pd.concat([train_y_good,train_y_bad])
            test_y = test_y_bad 

            pred_y,r2,rmse,mape = XGB(
                train_x.drop(columns=['name']),
                test_x.drop(columns=['name']),
                train_y,test_y)

            with open(output,'a') as results:
                results.write('=+'*25)
                results.write('\n Evaluation Metrics for CO2-to-CO Selectivity, S, for DS15 paper. (Mixed/Bad) Split = (80+80)/20')
                results.write('\n Number of MOFs used in Training: (Mixed/Bad)' + str(len(train_x)))
                results.write('\n List of MOFs used in Training: (Mixed/Bad)' + train_x['name'].to_string())
                results.write('\n' + '='*50)
                results.write('\n Number of MOFs used in Testing: (Mixed/Bad)' + str(len(test_x)))
                results.write('\n List of MOFs used in Training: (Mixed/Bad)' + test_x['name'].to_string())
                results.write('\n' + '='*50)
                results.write('\n Metrics of Evaluation (Mixed/Bad):')
                results.write('\n R2 Score: ' + str(r2))
                results.write('\n RMSE: ' + str(rmse))
                results.write('\n MAPE: ' + str(mape))
            
            mixed_bad.append(r2)
            mixed_bad.append(rmse)
            mixed_bad.append(mape)

            # Correlation Plot
            fig = plt.figure(figsize=(8,8),dpi=100)
            plt.rc('axes',axisbelow=True)
            plt.rcParams['font.size'] = 12
            plt.rcParams['font.family'] = 'DejaVu Sans'
            plt.grid(color = 'lightgrey',linestyle='-')

            max_y = 1
            if max(test_y)>=max(pred_y):
                max_y = max(test_y)
            else: 
                max_y = max(pred_y)
            
            plt.ylim(ymin=0,ymax=max_y+50)
            plt.xlim(xmin=0,xmax=max_y+50)
            plt.yticks(fontsize=8)
            plt.xticks(fontsize=8)

            x = np.arange(0,10000,1000)
            y = np.arange(0,10000,1000)
            plt.plot(x,y,color='violet',linestyle='--',alpha=0.50)
            plt.scatter(pred_y,test_y,color='purple',alpha=0.80)

            plt.xlabel('Predicted CO2-to-CO Selectivity, S')
            plt.ylabel('Real CO2-to-CO Selectivity, S')
            plt.title('Plot of Real vs Predicted CO2-to-CO Selectivty. Train=Mixed, Test=Bad')
            text = f'R2: {r2:.4f} \n RMSE: {rmse:.4f} \n MAPE: {mape:.3f} \n Train Size: {len(train_x)} \n Test Size: {len(test_x)}'
            plt.text(x=max_y*0.70,y=max_y*0.10,s=text)
            plt.savefig(r"C:\Users\Grace\Documents\Code\USRO 2025W\My Code\DS15_data\Correlation_Plots\DS15_mixed_bad.png")
            plt.close()

    with open(output,'a') as finals:
        finals.write('\n' + '#'*50)
        finals.write('\n Metrics of Evaluation. (MIXED/BAD)')
        finals.write('\n R2 (mixed/bad): ' + str(mixed_bad[0]))
        finals.write('\n RMSE (mixed/bad): ' + str(mixed_bad[1]))
        finals.write('\n MAPE (mixed/bad): ' + str(mixed_bad[2]))

        finals.write('\n Metrics of Evaluation. (MIXED/GOOD)')
        finals.write('\n R2 (mixed/good): ' + str(mixed_good[0]))
        finals.write('\n RMSE (mixed/good): ' + str(mixed_good[1]))
        finals.write('\n MAPE (mixed/good): ' + str(mixed_good[2]))

        finals.write('\n Metrics of Evaluation. (MIXED/MIXED)')
        finals.write('\n R2 (mixed/mixed): ' + str(mixed_mixed[0]))
        finals.write('\n RMSE (mixed/mixed): ' + str(mixed_mixed[1]))
        finals.write('\n MAPE (mixed/mixed): ' + str(mixed_mixed[2]))
