# =============================================================================
import numpy as np
import pandas as pd
import minepy as mp
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
# =============================================================================
# Functions:
# crossCorr(indicators)
# genMIC(x, y)
# loadMIC()
# mine_stats(mine)
# printMineStats(mine)
# =============================================================================
def crossCorr(indicators):
    Threshold = 0.85
    total_corr = indicators[list(indicators.keys())[0]].corr()
    total_corr[:] = 0
    for ticker in indicators:
        corr  = indicators[ticker].corr()        
        total_corr  = total_corr.add(corr, fill_value=0) 
    
    total_corr  = total_corr.divide(len(indicators.keys()))    
    corr_list  = []    
    for i in range(total_corr.shape[0]):
            for j in range(total_corr.shape[0]):
                if(i < j):
                    total_corr.iloc[i,j]  = -1
                elif(i > j):
                    if(total_corr.iloc[i,j]  > Threshold):                        
                        corr_list.append((total_corr.index[i]
                        + ':' + total_corr.columns[j]))
                        
    print('Highly Correlated Features (+0.85): ', corr_list, '\n')
    plt.figure()
    plt.title('Correlation Heat Map')
    sns.heatmap(total_corr, xticklabels=total_corr.columns.values,
                yticklabels=total_corr.columns.values, cmap='gray')
# =============================================================================    
def genMIC(x, y):
    mic  = {}
    mic_mean  = np.zeros((1, x[list(x.keys())[0]].shape[1]))
    
    mine = mp.MINE(alpha=0.6, c=15, est="mic_approx")
    
    for ticker in x:
        for i in range(x[ticker].shape[1]):
                
            feature_name = x[ticker].columns.values[i]
            if(i == 0):           
                mine.compute_score(x[ticker].iloc[:,i], y[ticker].iloc[:,0])
                mic[ticker] = mine_stats(mine)    
                mic[ticker].rename(columns={mic[ticker ].columns[0]: feature_name}, inplace=True)
            else:
                mine.compute_score(x[ticker].iloc[:,i], y[ticker].iloc[:,0])
                mic[ticker][feature_name] = mine_stats(mine)
                
        mic_mean  += mic[ticker]            
    mic_mean  /= len(x)
      
    #Dump Pickles
    pickle.dump(mic_mean, open('data/mic.pickle', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)    
# =============================================================================    
def loadMIC():
    mic = pickle.load(open('data/mic.pickle', 'rb'))
    return(mic)
# ============================================================================= 
def mine_stats(mine):
    mine_features = np.zeros(0)
    mine_features = np.append(mine_features, mine.mic())
    mine_features = pd.dataFrame(mine_features)
    return(mine_features)
# =============================================================================     
def printMineStats(mine):
    print( "MIC", mine.mic())
    print( "MAS", mine.mas())
    print( "MEV", mine.mev())
    print( "MCN (eps=0)", mine.mcn(0))
    print( "MCN (eps=1-MIC)", mine.mcn_general())
    print( "GMIC", mine.gmic())
    print( "TIC", mine.tic())
        

#%%