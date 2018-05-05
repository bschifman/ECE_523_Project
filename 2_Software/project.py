# =============================================================================
# Packages:

# Project Files:
import dataSetup as ds
import FeatureSelection as fs
import prediction as pred
# =============================================================================
# #%% Control Variables %%#
DEBUG           = False 
REINGEST_DATA   = False #imports data from quandl, dumps data to data.pickle
REGENERATE_TA   = True #recalc features, dumps to indicators_norm.pickle
REGENERATE_MIC  = False #recalc mic, dumps to mic.pickle 
PLOT_CORR       = False #calc corr, plot heat map
PREDICT         = False #run prediction algs.
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
    data = ds.ingestData()
    ds.dumpData(data, 'data')
else:
    data = ds.loadQdata()
# =============================================================================
#Regenerate feature pickle files
if(REGENERATE_TA):
    indicators_norm, y_norm, indicators, y = ds.genTA(data, t=T)
    
    #Dump Pickles
    ds.dumpData(indicators_norm,  'indicators_normT'+str(TIMEPERIODNUM))
    ds.dumpData(y_norm,  'y_normT'+str(TIMEPERIODNUM))
    ds.dumpData(indicators,  'indicatorsT'+str(TIMEPERIODNUM))
    ds.dumpData(y, 'yT'+str(TIMEPERIODNUM))
else: #Load data from pickles
    indicators_norm, y_norm, indicators, y = ds.loadTAdata(tNum=TIMEPERIODNUM)
    x_all, y_all = ds.reformat(indicators_norm, y_norm)
# =============================================================================    
if(PREDICT):
    pred.MLP(x_all, y_all)
# =============================================================================
#Regnerate mic pickle ***NEED TO FIX THE DIMENSIONS OF DATA AND Y IN ORDER TO MATCH***
if(REGENERATE_MIC):
<<<<<<< HEAD
    import time
    s = time.clock()
    mic = fs.genMIC(indicators_norm, y)
    print(time.clock() - s)
    
else:
    mic = fs.loadMIC()
# =============================================================================    
#Plot corr
if(PLOT_CORR):
    fs.crossCorr(indicators_norm)
    