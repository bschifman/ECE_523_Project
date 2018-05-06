# =============================================================================
# Packages:

# Project Files:
import dataSetup as ds
import featureSelection as fs
import prediction as pred
# =============================================================================
# #%% Control Variables %%#
DEBUG           = False 
REINGEST_DATA   = False #imports data from quandl, dumps data to data.pickle
REGENERATE_TA   = False #recalc features, dumps to indicators_norm.pickle
REGENERATE_MIC  = False #recalc mic, dumps to mic.pickle 
PLOT_CORR       = False #calc corr, plot heat map
PREDICT         = True #run prediction algs.
TIMEPERIODNUM    = 1
#Add more control here
# =============================================================================
if(TIMEPERIODNUM == 1):
    T = 10 #2 weeks
elif(TIMEPERIODNUM == 2):
    T = 25 #5 weeks
elif(TIMEPERIODNUM == 3):
    T = 5 #1 week
else:
    print('Pick a valid TIMEPERIODNUM')
    exit()
# =============================================================================
#Reimport data from quandl
if(REINGEST_DATA):
    data, y = ds.ingestData()
    ds.dumpData(data, 'data')
    ds.dumpData(y, 'y')
else:
    data, y = ds.loadQdata()
# =============================================================================
#Regenerate feature pickle files
if(REGENERATE_TA):
    indicators_norm, indicators, y_ind = ds.genTA(data, y, t=T)
    x_all, y_all = ds.reformat(indicators_norm, y_ind)
    
    #Dump Pickles
    ds.dumpData(indicators_norm,  'indicators_normT'+str(TIMEPERIODNUM))
    ds.dumpData(indicators,  'indicatorsT'+str(TIMEPERIODNUM))
    ds.dumpData(y_ind, 'y_indT'+str(TIMEPERIODNUM))
else: #Load data from pickles
    indicators_norm, indicators, y_ind = ds.loadTAdata(tNum=TIMEPERIODNUM)
    x_all, y_all = ds.reformat(indicators_norm, y_ind)
# =============================================================================
#Plot corr
if(PLOT_CORR):
    fs.crossCorr(indicators_norm, t=TIMEPERIODNUM)  
# =============================================================================
if(PREDICT):
#    pred.MLP(x_all, y_all)
#    pred.randomForest(x_all, y_all, switch=1, t=TIMEPERIODNUM)
    pred.pca(x_all, y_all, t=TIMEPERIODNUM)
# =============================================================================
#Regnerate mic pickle files
if(REGENERATE_MIC):
    mic = fs.genMIC(indicators, y_ind)
    ds.dumpData(mic, 'micT'+str(TIMEPERIODNUM))
#else: #Load mic from pickle
#    mic = fs.loadMIC(tNum=TIMEPERIODNUM)
# =============================================================================    

    