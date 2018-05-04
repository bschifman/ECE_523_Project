import numpy as np
import pandas as pd
import minepy as mp
import dataSetup as ds
import matplotlib.pyplot as plt
import time
import pickle
import seaborn as sns




def crossCorr(data_open_norm, data_high_norm, data_low_norm, data_close_norm):
    Threshold = 0.8
    total_open_corr = data_open_norm[list(data_open_norm.keys())[0]].corr()
    total_open_corr[:] = 0
    total_high_corr  = total_open_corr.copy()
    total_low_corr   = total_open_corr.copy()
    total_close_corr = total_open_corr.copy()
    for ticker in data_open_norm:
        corr_open  = data_open_norm[ticker].corr()
        corr_high  = data_high_norm[ticker].corr()
        corr_low   = data_low_norm[ticker].corr()
        corr_close = data_close_norm[ticker].corr()
        
        total_open_corr  = total_open_corr.add(corr_open, fill_value=0) 
        total_high_corr  = total_high_corr.add(corr_high, fill_value=0) 
        total_low_corr   = total_low_corr.add(corr_low, fill_value=0) 
        total_close_corr = total_close_corr.add(corr_close, fill_value=0) 
    
    total_open_corr  = total_open_corr.divide(len(data_open_norm.keys()))
    total_high_corr  = total_high_corr.divide(len(data_open_norm.keys()))
    total_low_corr   = total_low_corr.divide(len(data_open_norm.keys()))
    total_close_corr = total_close_corr.divide(len(data_open_norm.keys()))
    
    open_corr_list  = []
    high_corr_list  = []
    low_corr_list   = []
    close_corr_list = []
    for i in range(total_open_corr.shape[0]):
            for j in range(total_open_corr.shape[0]):
                if(i < j):
                    total_open_corr.iloc[i,j]  = -1
                    total_high_corr.iloc[i,j]  = -1
                    total_low_corr.iloc[i,j]   = -1
                    total_close_corr.iloc[i,j] = -1
                elif(i > j):
                    if(total_open_corr.iloc[i,j]  > Threshold):                        
                        open_corr_list.append((total_open_corr.index[i] + ':' + total_open_corr.columns[j]))
                    if(total_high_corr.iloc[i,j]  > Threshold):
                        high_corr_list.append((total_high_corr.index[i] + ':' + total_high_corr.columns[j]))
                    if(total_low_corr.iloc[i,j]   > Threshold):
                       low_corr_list.append((total_low_corr.index[i] + ':' + total_low_corr.columns[j]))
                    if(total_close_corr.iloc[i,j] > Threshold):
                        close_corr_list.append((total_close_corr.index[i] + ':' + total_close_corr.columns[j]))
    print('Correlated Open Features: ', open_corr_list, '\n')
    print('Correlated High Features: ', high_corr_list, '\n')
    print('Correlated Low Features: ', low_corr_list, '\n')
    print('Correlated Close Features: ', close_corr_list, '\n')     
                              
    plt.figure()
    plt.title('Correlation Open Heat Map')
    sns.heatmap(total_open_corr, xticklabels=total_open_corr.columns.values, yticklabels=total_open_corr.columns.values, cmap='gray')
    
    plt.figure()
    plt.title('Correlation High Heat Map')
    sns.heatmap(total_high_corr, xticklabels=total_high_corr.columns.values, yticklabels=total_high_corr.columns.values, cmap='gray')
    
    plt.figure()
    plt.title('Correlation Low Heat Map')
    sns.heatmap(total_low_corr, xticklabels=total_low_corr.columns.values, yticklabels=total_low_corr.columns.values, cmap='gray')
    
    plt.figure()
    plt.title('Correlation Close Heat Map')
    sns.heatmap(total_close_corr, xticklabels=total_close_corr.columns.values, yticklabels=total_close_corr.columns.values, cmap='gray')
    
    
#THIS TAKES 365seconds
def genMic(data, data_open, data_high, data_low, data_close, y_sign_daily):
#    start_time = time.clock()
    mic_open  = {}
    mic_high  = {}
    mic_low   = {}
    mic_close = {}    
    mic_open_mean  = np.zeros((1, data_open[list(data.keys())[0]].shape[1])) #size of a single mine_features matrix for OHLC
    mic_high_mean  = np.zeros((1, data_high[list(data.keys())[0]].shape[1])) #size of a single mine_features matrix for OHLC
    mic_low_mean   = np.zeros((1, data_low[list(data.keys())[0]].shape[1])) #size of a single mine_features matrix for OHLC
    mic_close_mean = np.zeros((1, data_close[list(data.keys())[0]].shape[1])) #size of a single mine_features matrix for OHLC
    
    mine = mp.MINE(alpha=0.6, c=15, est="mic_approx")
    
    for ticker in data:
        for i in range(data_open[ticker].shape[1]):
                
            feature_name = data_open[ticker].columns.values[i]
            if(i == 0):           
                mine.compute_score(data_open[ticker].iloc[:,i], y_sign_daily[ticker].iloc[:,0])
                mic_open[ticker] = mine_stats(mine)
                    
                mine.compute_score(data_high[ticker].iloc[:,i], y_sign_daily[ticker].iloc[:,1])
                mic_high[ticker] = mine_stats(mine)
                
                mine.compute_score(data_low[ticker].iloc[:,i], y_sign_daily[ticker].iloc[:,2])
                mic_low[ticker] = mine_stats(mine)
                
                mine.compute_score(data_close[ticker].iloc[:,i], y_sign_daily[ticker].iloc[:,3])
                mic_close[ticker] = mine_stats(mine)
                
                mic_open[ ticker].rename(columns={mic_open[ticker ].columns[0]: feature_name}, inplace=True)
                mic_high[ ticker].rename(columns={mic_high[ticker ].columns[0]: feature_name}, inplace=True)
                mic_low[  ticker].rename(columns={mic_low[ticker ].columns[0]: feature_name}, inplace=True)
                mic_close[ticker].rename(columns={mic_close[ticker ].columns[0]: feature_name}, inplace=True)
            else:
                mine.compute_score(data_open[ticker].iloc[:,i], y_sign_daily[ticker].iloc[:,0])
                mic_open[ticker][feature_name] = mine_stats(mine)
                        
                mine.compute_score(data_high[ticker].iloc[:,i], y_sign_daily[ticker].iloc[:,1])
                mic_high[ticker][feature_name] = mine_stats(mine)
                
                mine.compute_score(data_low[ticker].iloc[:,i], y_sign_daily[ticker].iloc[:,2])
                mic_low[ticker][feature_name] = mine_stats(mine)
                
                mine.compute_score(data_close[ticker].iloc[:,i], y_sign_daily[ticker].iloc[:,3])
                mic_close[ticker][feature_name] = mine_stats(mine)       
            
        mic_open_mean  += mic_open[ticker]  
        mic_high_mean  += mic_high[ticker]  
        mic_low_mean   += mic_low[ticker]   
        mic_close_mean += mic_close[ticker] 
           
    mic_open_mean  /= len(data)
    mic_high_mean  /= len(data)
    mic_low_mean   /= len(data)
    mic_close_mean /= len(data)
      
    #Dump Pickles
    pickle.dump(mic_open, open('data/mic_open.pickle', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)  
    pickle.dump(mic_high, open('data/mic_high.pickle', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)  
    pickle.dump(mic_low, open('data/mic_low.pickle', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)  
    pickle.dump(mic_close, open('data/mic_close.pickle', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    
    pickle.dump(mic_open_mean, open('data/mic_open_mean.pickle', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)    
    pickle.dump(mic_high_mean, open('data/mic_high_mean.pickle', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)  
    pickle.dump(mic_low_mean, open('data/mic_low_mean.pickle', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)  
    pickle.dump(mic_close_mean, open('data/mic_close_mean.pickle', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)  
    
# =============================================================================
#     end_time = time.clock()
#     total_time = end_time - start_time
#     print('Time: ', total_time)
# =============================================================================
def loadMic():
    mic_open         = pickle.load(open('data/mic_open.pickle', 'rb'))
    mic_high    = pickle.load(open('data/mic_high.pickle', 'rb'))
    mic_low    = pickle.load(open('data/mic_low.pickle', 'rb'))
    mic_close     = pickle.load(open('data/mic_close.pickle', 'rb'))
    
    mic_open_mu   = pickle.load(open('data/mic_open_mean.pickle', 'rb'))
    mic_high_mu = pickle.load(open('data/mic_high_mean.pickle', 'rb'))
    mic_low_mu   = pickle.load(open('data/mic_low_mean.pickle', 'rb'))
    mic_close_mu = pickle.load(open('data/mic_close_mean.pickle', 'rb'))
    
    return(mic_open, mic_high, mic_low, mic_close, mic_open_mu,
           mic_high_mu, mic_low_mu, mic_close_mu)
        
def print_mine_stats(mine):
    print( "MIC", mine.mic())
    print( "MAS", mine.mas())
    print( "MEV", mine.mev())
    print( "MCN (eps=0)", mine.mcn(0))
    print( "MCN (eps=1-MIC)", mine.mcn_general())
    print( "GMIC", mine.gmic())
    print( "TIC", mine.tic())
        
def mine_stats(mine):
    mine_features = np.zeros(0)
    mine_features = np.append(mine_features, mine.mic())
    mine_features = pd.DataFrame(mine_features)
#    mine_features = np.append(mine_features, mine.mas())
#    mine_features = np.append(mine_features, mine.mev())
#    mine_features = np.append(mine_features, mine.mcn(0))
#    mine_features = np.append(mine_features, mine.mcn_general())
#    mine_features = np.append(mine_features, mine.gmic())
#    mine_features = np.append(mine_features, mine.tic())    
    return(mine_features)


data_open_norm, data_high_norm, data_low_norm, data_close_norm, y = ds.loadTAdata(tNum=1)
crossCorr(data_open_norm, data_high_norm, data_low_norm, data_close_norm)
#genMic(data, data_open, data_high, data_low, data_close, y_sign_daily)
#mic_open, mic_high, mic_low, mic_close, mic_open_mu, mic_high_mu, mic_low_mu, mic_close_mu= loadMic()

#%%