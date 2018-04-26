    # =============================================================================
# Packages:

# Project Files:
import dataSetup as ds
import featureSelection as fs
# =============================================================================
# #%% Control Variables %%#
DEBUG         = True  #Maybe I'm stupid but I needed this to debug norm function
REINGEST_DATA = False #imports data from quandl, dumps data to data.pickle
REGENERATE_TA = False #recalc features on open, high, low, close ... dumps each to data_<x>.pickle
#Add more control here
# =============================================================================
#Reimport data from quandl
if(REINGEST_DATA):
    ds.ingestData()

#Regenerate feature pickle files
if(REGENERATE_TA):
    ds.genTA()

#Load data from pickles
data, data_open, data_high, data_low, data_close, y_sign_daily = ds.loadData()

#normData work in progress in dataSetup.py
if(DEBUG):
    data_close_norm5 = ds.normData(data_close, 5)