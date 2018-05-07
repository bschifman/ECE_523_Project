# =============================================================================
# Packages:
from keras.utils import np_utils

# Project Files:
import dataSetup as ds
import FeatureSelection as fs
import prediction as pred
# =============================================================================
# #%% Control Variables %%#
DEBUG           = False 
REINGEST_DATA   = False #imports data from quandl, dumps data to data.pickle
LOAD_DATA       = False 
REGENERATE_TA   = False #recalc features, dumps to indicators_norm.pickle
LOAD_TA         = False
REGENERATE_MIC  = False  #recalc mic, dumps to mic.pickle 
LOAD_MIC        = False
PLOT_CORR       = False #calc corr, plot heat map
PREDICT         = False #run prediction algs.
RUN_MLP         = False
RUN_RFE         = False
PLOT_RFE        = False
PLOT_MLP        = True
TIMEPERIODNUM    = 1
#Add more control here
# =============================================================================
if(TIMEPERIODNUM == 1):
    T = 10 #2 weeks
else:
    print('Pick a valid TIMEPERIODNUM')
    exit()
# =============================================================================
#Reimport data from quandl
if(REINGEST_DATA):
    data, y, dataTEST, yTEST = ds.ingestData()
    ds.dumpData(data, 'data')
    ds.dumpData(y, 'y')
    ds.dumpData(dataTEST, 'dataTEST')
    ds.dumpData(yTEST, 'yTEST')
if(LOAD_DATA):
    data, y, dataTEST, yTEST = ds.loadQdata()
# =============================================================================
#Regenerate feature pickle files
if(REGENERATE_TA):
    indicators_norm, indicators, y_ind = ds.genTA(data, y, t=T)
    xTestNorm, xTest, yTest = ds.genTA(dataTEST, yTEST, t=T)
    x_all, y_all = ds.reformat(indicators_norm, y_ind)
    x_test, y_test = ds.reformat(xTestNorm, yTest)
    
    #Dump Pickles
    ds.dumpData(indicators_norm,  'indicators_normT'+str(TIMEPERIODNUM))
    ds.dumpData(indicators,  'indicatorsT'+str(TIMEPERIODNUM))
    ds.dumpData(y_ind, 'y_indT'+str(TIMEPERIODNUM))
    ds.dumpData(xTestNorm,  'indicators_normT'+str(TIMEPERIODNUM)+'TEST')
    ds.dumpData(xTest,  'indicatorsT'+str(TIMEPERIODNUM)+'TEST')
    ds.dumpData(yTest, 'y_indT'+str(TIMEPERIODNUM)+'TEST')
if(LOAD_TA): #Load data from pickles
    indicators_norm, indicators, y_ind, xTestNorm, xTest, yTest = ds.loadTAdata(tNum=TIMEPERIODNUM)
    x_all, y_all = ds.reformat(indicators_norm, y_ind)
    x_test, y_test = ds.reformat(xTestNorm, yTest)
# =============================================================================
#Plot corr
if(PLOT_CORR):
    fs.crossCorr(indicators_norm, t=TIMEPERIODNUM)  
# =============================================================================
if(RUN_RFE):
    #Normalized
    selSVM5, selF_SVM5_Acc = fs.RFE_SVM(x_all, y_all, 5)
    selSVM10, selF_SVM10_Acc = fs.RFE_SVM(x_all, y_all, 10)
    selSVM15, selF_SVM15_Acc = fs.RFE_SVM(x_all, y_all, 15)
    
    selAda5, selF_Ada5_Acc = fs.RFE_AdaBoost(x_all, y_all, 5)
    selAda10, selF_Ada10_Acc = fs.RFE_AdaBoost(x_all, y_all, 10)
    selAda15, selF_Ada15_Acc = fs.RFE_AdaBoost(x_all, y_all, 15)
if(PLOT_RFE):
    selF_SVM5_Acc2 = selSVM5.score(x_test, y_test)
    selF_SVM10_Acc2 = selSVM10.score(x_test, y_test)
    selF_SVM15_Acc2 = selSVM15.score(x_test, y_test)
    selF_Ada5_Acc2 = selAda5.score(x_test, y_test)
    selF_Ada10_Acc2 = selAda10.score(x_test, y_test)
    selF_Ada15_Acc2 = selAda15.score(x_test, y_test)
    
    rfeSvmAcc1 = [selF_SVM5_Acc, selF_SVM10_Acc, selF_SVM15_Acc]
    rfeSvmAcc2 = [selF_SVM5_Acc2, selF_SVM10_Acc2, selF_SVM15_Acc2]
    rfeAdaAcc1 = [selF_Ada5_Acc, selF_Ada10_Acc, selF_Ada15_Acc]
    rfeAdaAcc2 = [selF_Ada5_Acc2, selF_Ada10_Acc2, selF_Ada15_Acc2]
    fs.plotRFE(rfeSvmAcc1, rfeSvmAcc2, rfeAdaAcc1, rfeAdaAcc2)
if(RUN_MLP):
    y_test_keras = np_utils.to_categorical(y_test, 3)
    #All Features
    mlp, mlpAcc = pred.MLP(x_all, y_all)
    mlpAcc2 = mlp.evaluate(x_test, y_test_keras)[1]
    #Handpicked
    x_hand = x_all[['RSI', 'MACD', 'ATR', 'SAR', 'OBV']].copy()
    x_handTest = x_test[['RSI', 'MACD', 'ATR', 'SAR', 'OBV']].copy()
    mlpHand, mlpHandAcc = pred.MLP(x_hand, y_all)
    mlpHandAcc2 = mlpHand.evaluate(x_handTest, y_test_keras)[1]
    #RFE_Ada with 5 features selected
    x_ada5 = x_all[['ROC', 'MACD', 'OBV', 'ATR', 'VAR']].copy()
    x_ada5Test = x_test[['ROC', 'MACD', 'OBV', 'ATR', 'VAR']].copy()
    mlpAda5, mlpAda5Acc = pred.MLP(x_ada5, y_all)
    mlpAda5Acc2 = mlpAda5.evaluate(x_ada5Test, y_test_keras)[1]
if(PLOT_MLP):    
    fs.plotMLP()
if(PREDICT):
#    pred.randomForest(x_all, y_all, switch=1, t=TIMEPERIODNUM)
    pred.pca(x_all, y_all, t=TIMEPERIODNUM)
# =============================================================================
#Regnerate mic pickle files
if(REGENERATE_MIC):
    mic = fs.genMIC(indicators, y_ind, t=TIMEPERIODNUM)
    ds.dumpData(mic, 'MICT'+str(TIMEPERIODNUM))
if(LOAD_MIC): #Load mic from pickle
    mic = fs.loadMIC(tNum=TIMEPERIODNUM)
#=============================================================================    

