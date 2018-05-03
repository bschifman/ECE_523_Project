# =============================================================================
# Packages:

# Project Files:
import dataSetup as ds
import FeatureSelection as fs
# =============================================================================
# #%% Control Variables %%#
DEBUG         = False #Maybe I'm stupid but I needed this to debug norm function
REINGEST_DATA = False #imports data from quandl, dumps data to data.pickle
REGENERATE_TA = False #recalc features on open, high, low, close ... dumps each to data_<x>.pickle
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

#Reimport data from quandl
if(REINGEST_DATA):
    data = ds.ingestData()
    ds.dumpData(data, 'data')
else:
    data = ds.loadQdata()

#Regenerate feature pickle files
if(REGENERATE_TA):
    data_open_norm, data_high_norm, data_low_norm, data_close_norm, y = ds.genTA(data, t=T)
        
    #Dump Pickles
    ds.dumpData(data_open_norm,  'data_open_normT' +str(TIMEPERIODNUM))
    ds.dumpData(data_high_norm,  'data_high_normT' +str(TIMEPERIODNUM))
    ds.dumpData(data_low_norm,   'data_low_normT'  +str(TIMEPERIODNUM))
    ds.dumpData(data_close_norm, 'data_close_normT'+str(TIMEPERIODNUM))
    ds.dumpData(y, 'y')
else: #Load data from pickles
    data_open_norm, data_high_norm, data_low_norm, data_close_norm, y = ds.loadTAdata(tNum=TIMEPERIODNUM)
    x_open, x_high, x_low, x_close, y_all = ds.reformat(data_open_norm, data_high_norm, data_low_norm, data_close_norm, y)
#if(DEBUG):
