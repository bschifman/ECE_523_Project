# =============================================================================
# Packages:
import quandl
import pandas as pd
import numpy as np
import talib as ta
import pickle
quandl.ApiConfig.api_key = 'HwQoB4ePcDi8bFzJ6SJA'
# =============================================================================
# Functions:
# ingestData()
# genTA()
# loadData()
# normData()
# =============================================================================
# Generate Pickle #
def ingestData():
    start_date = '2006-12-31'
    end_date = '2017-12-31'
    database_code = 'WIKI'
    
    #tickers_financials = []
    #tickers_utilities = []
    #tickers_energy = []
    #tickers_healthcare = []
    #tickers_technology = []
    #tickers_realestate = []
    
    #THIS TAKES ABOUT A MINUTE WITH THE CURRENT DATES
    tickers_financials = ['JPM', 'BAC', 'WFC', 'C', 'MS']
    tickers_utilities = ['T', 'VZ', 'NEE', 'TMUS']
    tickers_energy = ['XOM', 'CVX', 'BP', 'GE', 'SLB']
    tickers_healthcare = ['JNJ', 'UNH', 'PFE', 'MRK', 'ABBV', 'MMM', 'AMGN', 'MDT']
    tickers_technology = ['AAPL', 'GOOGL', 'MSFT', 'FB', 'INTC', 'CSCO', 'ORCL', 'IBM', 'NVDA']
    tickers_realestate = ['ECL', 'DWDP', 'FMC', 'IP', 'PPG', 'VMC', 'BMS']
    tickers = [tickers_financials, tickers_utilities, tickers_energy, tickers_healthcare, tickers_technology, tickers_realestate]
    
    temp = {'Financials': tickers_financials,
        'Utilities': tickers_utilities,
        'Energy': tickers_energy,
        'Healthcare':tickers_healthcare,
        'Technology':tickers_technology,
        'Real Estate': tickers_realestate}
    pd.DataFrame.from_dict(temp, orient='index').transpose().to_csv('../3_Deliverables/Final Paper/tickers.csv', index=False)
    
    data = {}
    
    #data = quandl.get('WIKI/' + tickers, start_date=start_date, end_date=end_date)
    
    def import_data(ticker, start_date, end_date):
        return quandl.get('WIKI/' + ticker, start_date=start_date, end_date=end_date)
    
    for index in tickers:
        for ticker in index:
            print(ticker)
            data[ticker] = import_data(ticker, start_date, end_date)
    #REFORMAT DATA 
    for ticker in data:
        data[ticker].drop(['Open', 'High', 'Low', 'Close', 'Volume', 'Ex-Dividend', 'Split Ratio'], axis=1, inplace=True)
        data[ticker].rename(index=str, columns={"Adj. Open": "Open", "Adj. High": "High", "Adj. Low": "Low", "Adj. Close":
            "Close", "Adj. Volume": "Volume"}, inplace=True)
        
    return data
# =============================================================================
def genTA(data, t): #t is timeperiod
    data_open  = {}
    data_high  = {}
    data_low   = {}
    data_close = {}
    
    #labels
    sign_daily = {}
   
    for ticker in data:
        data_open[ticker ] = ta.SMA(data[ticker].iloc[:,0], timeperiod=t).to_frame()
        data_high[ticker ] = ta.SMA(data[ticker].iloc[:,1], timeperiod=t).to_frame()
        data_low[ticker  ] = ta.SMA(data[ticker].iloc[:,2], timeperiod=t).to_frame()
        data_close[ticker] = ta.SMA(data[ticker].iloc[:,3], timeperiod=t).to_frame()
    
        data_open[ticker ]['RSI'] = ta.RSI(data[ticker].iloc[:,0], timeperiod=(t-1))
        data_high[ticker ]['RSI'] = ta.RSI(data[ticker].iloc[:,1], timeperiod=(t-1))
        data_low[ticker  ]['RSI'] = ta.RSI(data[ticker].iloc[:,2], timeperiod=(t-1))
        data_close[ticker]['RSI'] = ta.RSI(data[ticker].iloc[:,3], timeperiod=(t-1))
    
        data_open[ticker ]['OBV'] = ta.OBV(data[ticker].iloc[:,0], data[ticker].iloc[:,4])
        data_high[ticker ]['OBV'] = ta.OBV(data[ticker].iloc[:,1], data[ticker].iloc[:,4])
        data_low[ticker  ]['OBV'] = ta.OBV(data[ticker].iloc[:,2], data[ticker].iloc[:,4])
        data_close[ticker]['OBV'] = ta.OBV(data[ticker].iloc[:,3], data[ticker].iloc[:,4])
    
        data_open[ticker ]['EMA'] = ta.EMA(data[ticker].iloc[:,0], timeperiod=t)
        data_high[ticker ]['EMA'] = ta.EMA(data[ticker].iloc[:,1], timeperiod=t)
        data_low[ticker  ]['EMA'] = ta.EMA(data[ticker].iloc[:,2], timeperiod=t)
        data_close[ticker]['EMA'] = ta.EMA(data[ticker].iloc[:,3], timeperiod=t)
    
        data_open[ticker ]['BBAND_Upper'], data_open[ticker ]['BBAND_Middle' ], data_open[ticker ]['BBAND_Lower' ] = ta.BBANDS(data[ticker].iloc[:,0], timeperiod=t, nbdevup=2, nbdevdn=2, matype=0)
        data_high[ticker ]['BBAND_Upper'], data_high[ticker ]['BBAND_Middle' ], data_high[ticker ]['BBAND_Lower' ] = ta.BBANDS(data[ticker].iloc[:,1], timeperiod=t, nbdevup=2, nbdevdn=2, matype=0)
        data_low[ticker  ]['BBAND_Upper'], data_low[ticker  ]['BBAND_Middle' ], data_low[ticker  ]['BBAND_Lower' ] = ta.BBANDS(data[ticker].iloc[:,2], timeperiod=t, nbdevup=2, nbdevdn=2, matype=0)
        data_close[ticker]['BBAND_Upper'], data_close[ticker]['BBAND_Middle' ], data_close[ticker]['BBAND_Lower' ] = ta.BBANDS(data[ticker].iloc[:,3], timeperiod=t, nbdevup=2, nbdevdn=2, matype=0)  
    
        data_open[ticker ]['ATR'] = ta.ATR(data[ticker].iloc[:,1], data[ticker].iloc[:,2], data[ticker].iloc[:,3],  timeperiod=(t-1))
        data_high[ticker ]['ATR'] = ta.ATR(data[ticker].iloc[:,1], data[ticker].iloc[:,2], data[ticker].iloc[:,3],  timeperiod=(t-1))
        data_low[ticker  ]['ATR'] = ta.ATR(data[ticker].iloc[:,1], data[ticker].iloc[:,2], data[ticker].iloc[:,3],  timeperiod=(t-1))
        data_close[ticker]['ATR'] = ta.ATR(data[ticker].iloc[:,1], data[ticker].iloc[:,2], data[ticker].iloc[:,3],  timeperiod=(t-1))
        
        data_open[ticker ]['MOM'] = ta.MOM(data[ticker].iloc[:,0], timeperiod=(t-1))
        data_high[ticker ]['MOM'] = ta.MOM(data[ticker].iloc[:,1], timeperiod=(t-1))
        data_low[ticker  ]['MOM'] = ta.MOM(data[ticker].iloc[:,2], timeperiod=(t-1))
        data_close[ticker]['MOM'] = ta.MOM(data[ticker].iloc[:,3], timeperiod=(t-1))
        
        data_open[ticker ].rename(columns={data_open[ticker ].columns[0]: "SMA"}, inplace=True)
        data_high[ticker ].rename(columns={data_high[ticker ].columns[0]: "SMA"}, inplace=True)
        data_low[ticker  ].rename(columns={data_low[ticker  ].columns[0]: "SMA"}, inplace=True)
        data_close[ticker].rename(columns={data_close[ticker].columns[0]: "SMA"}, inplace=True)
        
        #Daily Labels
        sign_daily[ticker]           = pd.DataFrame(np.concatenate((np.array([0.0]),np.sign(np.diff(data[ticker].iloc[:,0])))))
        sign_daily[ticker].rename(columns={sign_daily[ticker ].columns[0]: "Open"}, inplace=True)
        sign_daily[ticker]['High']   = pd.DataFrame(np.concatenate((np.array([0.0]),np.sign(np.diff(data[ticker].iloc[:,1])))))
        sign_daily[ticker]['Low']    = pd.DataFrame(np.concatenate((np.array([0.0]),np.sign(np.diff(data[ticker].iloc[:,2])))))
        sign_daily[ticker]['Close']  = pd.DataFrame(np.concatenate((np.array([0.0]),np.sign(np.diff(data[ticker].iloc[:,3])))))
        
        #drop 'nan' values
        data_open[ticker ].drop(data_open[ticker ].index[np.arange(0,t-1)], inplace=True)
        data_high[ticker ].drop(data_high[ticker ].index[np.arange(0,t-1)], inplace=True)
        data_low[ticker  ].drop(data_low[ticker  ].index[np.arange(0,t-1)], inplace=True)
        data_close[ticker].drop(data_close[ticker].index[np.arange(0,t-1)], inplace=True)
        
    norm_timeperiod = 60
    data_open_norm  = normData(data_open , norm_timeperiod)
    data_high_norm  = normData(data_high , norm_timeperiod)
    data_low_norm   = normData(data_low  , norm_timeperiod)
    data_close_norm = normData(data_close, norm_timeperiod)
    
    for ticker in sign_daily:
        N = np.size(sign_daily[ticker],0) - np.size(data_open_norm[ticker],0)
        sign_daily[ticker].drop(sign_daily[ticker].index[np.arange(0,N)], inplace=True)
        
    return data_open_norm, data_high_norm, data_low_norm, data_close_norm, sign_daily

# =============================================================================
def loadQdata():
    data = pickle.load(open('data/data.pickle', 'rb'))
    return data
# =============================================================================
def loadTAdata(tNum): # tNum = 1 or 2 or 3 (int)
    if(tNum != 1 and tNum != 2 and tNum != 3):
        print('loadData:tNum must be integer 1, 2, or 3')
        exit()
        
    tNum_str = str(tNum)
        
    data_open    = pickle.load(open('data/data_open_normT'+tNum_str+'.pickle', 'rb'))
    data_high    = pickle.load(open('data/data_high_normT'+tNum_str+'.pickle', 'rb'))
    data_low     = pickle.load(open('data/data_low_normT'+tNum_str+'.pickle', 'rb'))
    data_close   = pickle.load(open('data/data_close_normT'+tNum_str+'.pickle', 'rb'))
    y            = pickle.load(open('data/y.pickle', 'rb'))
    
    featureNames = {'Indicators':list(data_close[list(data_close.keys())[0]].columns.values)}
    pd.DataFrame.from_dict(featureNames).to_csv('../3_Deliverables/Final Paper/features.csv', index=False)
    
    return(data_open, data_high, data_low, data_close, y)
# =============================================================================
def dumpData(data, name_str): #name_str = (str) name of pickle file, ex: name_str='data_open_normT1'
    pickle.dump(data,  open('data/'+name_str+'.pickle',  'wb'), protocol=pickle.HIGHEST_PROTOCOL)
# =============================================================================
# Regular Normalization depends on a time period. norm_window = number of indices in period
# if length of dataframes don't divide evenly by timeperiod, throws out remainder from the beginning
def normData(dataIn, norm_window):
    dataOut = dataIn
    numCols = np.size(dataIn[list(dataIn.keys())[0]].keys())
    for ticker in dataIn:
        N = np.size(dataIn[ticker],0)
        r = N%norm_window
        rInd = np.arange(0,r)
        dataOut[ticker].iloc[rInd,:] = np.tile(float('nan'),(r,numCols))
        pStartInd = np.arange(r,N,norm_window)
        for i in pStartInd:
            timeInd = np.arange(i,i+norm_window)
            tempMin = np.min(dataIn[ticker].iloc[timeInd,:], axis=0)
            tempMax = np.max(dataIn[ticker].iloc[timeInd,:], axis=0)
            dataOut[ticker].iloc[timeInd,:] = np.divide(np.array((dataIn[ticker].iloc[timeInd,:] - tempMin)),np.array((tempMax - tempMin)))
        dataOut[ticker ].drop(dataOut[ticker ].index[rInd], inplace=True)
        
    return dataOut
# =============================================================================
def reformat(data_open, data_high, data_low, data_close, y):
    data_open_new  = pd.DataFrame()
    data_high_new  = pd.DataFrame()
    data_low_new   = pd.DataFrame()
    data_close_new = pd.DataFrame()
    y_new          = pd.DataFrame()
    for ticker in y:
        data_open_new  = data_open_new.append(data_open[ticker])
        data_high_new  = data_high_new.append(data_high[ticker])
        data_low_new   = data_low_new.append(data_low[ticker])
        data_close_new = data_close_new.append(data_close[ticker])
        y_new          = y_new.append(y[ticker])
    return data_open_new, data_high_new, data_low_new, data_close_new, y_new
    