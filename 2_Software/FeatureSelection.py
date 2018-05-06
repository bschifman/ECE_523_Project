# =============================================================================
import numpy as np
import pandas as pd
import minepy as mp
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
import time
# =============================================================================
# Functions:
# crossCorr(indicators, tNum)
# genMIC(x, y)
# loadMIC(tNum)
# mine_stats(mine)
# printMineStats(mine)
# RFE_SVM(x, y)
# RFE_AdaBoost(x, y)
# =============================================================================
def crossCorr(indicators, t):
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
                        
#    print('\n Highly Correlated Features (+0.85): ', corr_list, '\n')
    plt.figure(figsize=(22.0, 14.0))
#    plt.title('Correlation Heat Map')
    sns_plot = sns.heatmap(total_corr, xticklabels=total_corr.columns.values,
                yticklabels=total_corr.columns.values, cmap='gray')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    fig = sns_plot.get_figure()
    fig.savefig('../3_Deliverables/Final Paper/data/heatmapT'+str(t)+'.png')
    total_corr.to_csv('../3_Deliverables/Final Paper/data/corrT'+str(t)+'.csv')    
# =============================================================================    
def genMIC(x, y, t):
    mic  = {}
    mic_mean  = np.zeros((1, x[list(x.keys())[0]].shape[1]))
    f_names = list(x[list(x.keys())[0]].columns.values)
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
    
    mic_mean = pd.DataFrame(mic_mean) 
    mic_mean.columns = f_names
    
    plot = mic_mean.plot.bar(figsize=(22.0, 14.0))
    plot.set_xlabel('Features', fontsize=24)
    plot.set_ylabel('MIC', fontsize=24)
    plot.tick_params(labelsize=24)
    fig  = plot.get_figure()
    fig.savefig('../3_Deliverables/Final Paper/data/MICT'+str(t)+'.png')

    return(mic_mean)
# =============================================================================    
def loadMIC(tNum):
    if(tNum != 1 and tNum != 2 and tNum != 3):
        print('loadMIC:tNum must be integer 1, 2, or 3')
        exit()
    tNum_str = str(tNum)
    mic = pickle.load(open('data/MICT'+tNum_str+'.pickle', 'rb'))
    return(mic)    
# ============================================================================= 
def mine_stats(mine):
    mine_features = np.zeros(0)
    mine_features = np.append(mine_features, mine.mic())
    mine_features = pd.DataFrame(mine_features)
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
    
# =============================================================================
def RFE_SVM(x, y, numFeats):
    np.random.seed(7)
    
    x_train,  x_test,  y_train,  y_test  = train_test_split(x, y, test_size=0.33)
    
    clf = SVC(kernel='linear',probability=True, max_iter=200)
    selector = RFE(estimator=clf, n_features_to_select=numFeats, step=5)
    
    s_time = time.clock()
    
    selector.fit(x_train, y_train.iloc[:,0])
    
    acc = selector.score(x_test, y_test.iloc[:,0])
    
    e_time = time.clock()
    print('\n Total Time: ', e_time-s_time)
    print(' Accuracy:', acc)
    return selector, acc
# =============================================================================
def RFE_AdaBoost(x, y, numFeats):
    np.random.seed(7)
    
    x_train,  x_test,  y_train,  y_test  = train_test_split(x, y, test_size=0.33)
    
    clf = AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=1.0)
    selector = RFE(estimator=clf, n_features_to_select=numFeats, step=5)
    
    s_time = time.clock()
    
    selector.fit(x_train, y_train.iloc[:,0])
    
    acc = selector.score(x_test, y_test.iloc[:,0])
    
    e_time = time.clock()
    print('\n Total Time: ', e_time-s_time)
    print(' Accuracy:', acc)
    return selector, acc
# =============================================================================
def plotRFE(rfeSvmAcc1, rfeSvmAcc2, rfeAdaAcc1, rfeAdaAcc2):
    numFeats = [5,10,15]
    fig, ax1 = plt.subplots(figsize=(22.0, 14.0)) 
    plt.plot(numFeats, rfeSvmAcc1, '--o', label='SVM1')
    plt.plot(numFeats, rfeSvmAcc2, '--o', label='SVM2')
    plt.plot(numFeats, rfeAdaAcc1, '--o', label='Ada1')
    plt.plot(numFeats, rfeAdaAcc2, '--o', label='Ada2')
    plt.legend(fontsize=24)
    plt.xlabel('Number of Selected Features', fontsize=24)
    plt.ylabel('Accuracy', fontsize=24)
    plt.tick_params(labelsize=24)
    fig.savefig('../3_Deliverables/Final Paper/data/RFE.png')