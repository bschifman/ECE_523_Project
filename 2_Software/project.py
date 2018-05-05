# =============================================================================
# Packages:

# Project Files:
import dataSetup as ds
#import FeatureSelection as fs
# =============================================================================
# #%% Control Variables %%#
DEBUG         = False 
REINGEST_DATA = False #imports data from quandl, dumps data to data.pickle
REGENERATE_TA = False #recalc features, dumps to indicators_norm.pickle
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

#Reimport data from quandl
if(REINGEST_DATA):
    data = ds.ingestData()
    ds.dumpData(data, 'data')
else:
    data = ds.loadQdata()

#Regenerate feature pickle files
if(REGENERATE_TA):
    indicators_norm, y = ds.genTA(data, t=T)
        
    #Dump Pickles
    ds.dumpData(indicators_norm,  'indicators_normT' +str(TIMEPERIODNUM))
    ds.dumpData(y, 'y')
else: #Load data from pickles
    indicators_norm, y = ds.loadTAdata(tNum=TIMEPERIODNUM)
    x_all, y_all = ds.reformat(indicators_norm, y)
