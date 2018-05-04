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
PREDICT         = False #run prediction algs.
TIMEPERIODNUM    = 3
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
    indicators_norm, y = ds.genTA(data, t=T)
    x_all, y_all = ds.reformat(indicators_norm, y)
    
    #Dump Pickles
    ds.dumpData(indicators_norm,  'indicators_normT' +str(TIMEPERIODNUM))
    ds.dumpData(y, 'y')
else: #Load data from pickles
    indicators_norm, y = ds.loadTAdata(tNum=TIMEPERIODNUM)
    x_all, y_all = ds.reformat(indicators_norm, y)
# =============================================================================    
if(PREDICT):
    pred.MLP(x_all, y_all)
# =============================================================================
#Regnerate mic pickle ***NEED TO FIX THE DIMENSIONS OF DATA AND Y IN ORDER TO MATCH***
#if(REGENERATE_MIC):
#    mic = fs.genMic(data, y)
#    
#else:
#    mic = fs.loadMic()
# =============================================================================    
#Plot corr
if(PLOT_CORR):
    fs.crossCorr(indicators_norm)
    