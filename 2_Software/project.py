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
RUN_MLP         = True
RUN_RFE         = False
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
if(RUN_MLP):
    mlp, mplAcc = pred.MLP(x_all, y_all)
    y_test_keras = np_utils.to_categorical(y_test, 3)
    mlpAcc2 = mlp.evaluate(x_test, y_test_keras)[1]
if(RUN_RFE):
    #Normalized
    selSVM, selF_SVM_Acc = fs.RFE_SVM(x_all, y_all, 5)
    selAda, selF_Ada_Acc = fs.RFE_AdaBoost(x_all, y_all, 5)
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

