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
# reformat()
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
    pd.DataFrame.from_dict(temp, orient='index').transpose().to_csv('../3_Deliverables/Final Paper/data/tickers.csv', index=False)
    
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
    indicators  = {}
    
    #labels
    sign_daily = {}
   
    for ticker in data:
    ## Overlap
        indicators[ticker] = ta.SMA(data[ticker].iloc[:,3], timeperiod=t).to_frame()        
        indicators[ticker]['EMA'] = ta.EMA(data[ticker].iloc[:,3], timeperiod=t)       
        indicators[ticker]['BBAND_Upper'], indicators[ticker]['BBAND_Middle' ], indicators[ticker]['BBAND_Lower' ] = ta.BBANDS(data[ticker].iloc[:,3], timeperiod=t, nbdevup=2, nbdevdn=2, matype=0)         
        indicators[ticker]['HT_TRENDLINE'] = ta.HT_TRENDLINE(data[ticker].iloc[:,3])
        indicators[ticker]['SAR'] = ta.SAR(data[ticker].iloc[:,1], data[ticker].iloc[:,2], acceleration=0, maximum=0)
        #rename SMA column
        indicators[ticker].rename(columns={indicators[ticker].columns[0]: "SMA"}, inplace=True)
    ## Momentum
        indicators[ticker]['RSI'] = ta.RSI(data[ticker].iloc[:,3], timeperiod=(t-1))
        indicators[ticker]['MOM'] = ta.MOM(data[ticker].iloc[:,3], timeperiod=(t-1))
        indicators[ticker]['ROC'] = ta.ROC(data[ticker].iloc[:,3], timeperiod=(t))
        indicators[ticker]['ROCP']= ta.ROCP(data[ticker].iloc[:,3],timeperiod=(t))
        #Skipping STOCH for now needs slow and fast period
        #Skipping MACD for now needs slow, fast, and signal period
        
    ## Volume
        indicators[ticker]['OBV'] = ta.OBV(data[ticker].iloc[:,3], data[ticker].iloc[:,4])
        indicators[ticker]['AD'] = ta.AD(data[ticker].iloc[:,1], data[ticker].iloc[:,2], data[ticker].iloc[:,3], data[ticker].iloc[:,4])
        #Skipping ADOSC for now needs slow and fast period
        
    ## Cycle
        indicators[ticker]['HT_DCPERIOD'] = ta.HT_DCPERIOD(data[ticker].iloc[:,3])
        indicators[ticker]['HT_TRENDMODE']= ta.HT_TRENDMODE(data[ticker].iloc[:,3])
    
    ## Price
        indicators[ticker]['AVGPRICE'] = ta.AVGPRICE(data[ticker].iloc[:,0], data[ticker].iloc[:,1], data[ticker].iloc[:,2], data[ticker].iloc[:,3])
        indicators[ticker]['TYPPRICE'] = ta.TYPPRICE(data[ticker].iloc[:,1], data[ticker].iloc[:,2], data[ticker].iloc[:,3])
    
    ## Volatility
        indicators[ticker]['ATR'] = ta.ATR(data[ticker].iloc[:,1], data[ticker].iloc[:,2], data[ticker].iloc[:,3],  timeperiod=(t-1))
    
    ## Statistics
        indicators[ticker]['BETA'] = ta.BETA(data[ticker].iloc[:,1], data[ticker].iloc[:,2], timeperiod=t)
        indicators[ticker]['LINEARREG'] = ta.LINEARREG(data[ticker].iloc[:,3], timeperiod=t)
        indicators[ticker]['VAR'] = ta.VAR(data[ticker].iloc[:,3], timeperiod=t, nbdev=1)
    
    ## Math Transform
        indicators[ticker]['EXP'] = ta.EXP(data[ticker].iloc[:,3])
        indicators[ticker]['LN'] = ta.LN(data[ticker].iloc[:,3])
    
    ## Patterns (returns integers - but norming might not really do anything but wondering if they should be normed)
        indicators[ticker]['CDLBREAKAWAY'] = ta.CDLBREAKAWAY(data[ticker].iloc[:,0], data[ticker].iloc[:,1], data[ticker].iloc[:,2], data[ticker].iloc[:,3])
        indicators[ticker]['CDLDOJI']      = ta.CDLDOJI(data[ticker].iloc[:,0], data[ticker].iloc[:,1], data[ticker].iloc[:,2], data[ticker].iloc[:,3])
        indicators[ticker]['CDLHAMMER']    = ta.CDLHAMMER(data[ticker].iloc[:,0], data[ticker].iloc[:,1], data[ticker].iloc[:,2], data[ticker].iloc[:,3])
        indicators[ticker]['CDLHANGINGMAN']= ta.CDLHANGINGMAN(data[ticker].iloc[:,0], data[ticker].iloc[:,1], data[ticker].iloc[:,2], data[ticker].iloc[:,3])
        
    #Daily Labels
        sign_daily[ticker] = pd.DataFrame(np.concatenate((np.array([0.0]),np.sign(np.diff(data[ticker].iloc[:,3])))))
        
    #drop 'nan' values
        indicators[ticker].drop(indicators[ticker].index[np.arange(0,t-1)], inplace=True)
        
    #Normalize Features
    norm_timeperiod = 60 #Normalize over 3 month windows
    indicators_norm = normData(indicators, norm_timeperiod)
    
    for ticker in sign_daily:
        N = np.size(sign_daily[ticker],0) - np.size(indicators_norm[ticker],0)
        sign_daily[ticker].drop(sign_daily[ticker].index[np.arange(0,N)], inplace=True)
        
    return indicators_norm, sign_daily

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
        
    indicators   = pickle.load(open('data/indicators_normT'+tNum_str+'.pickle', 'rb'))
    y            = pickle.load(open('data/y.pickle', 'rb'))
    
    featureNames = {'Indicators':list(indicators[list(indicators.keys())[0]].columns.values)}
    pd.DataFrame.from_dict(featureNames).to_csv('../3_Deliverables/Final Paper/data/features.csv', index=False)
    
    return(indicators, y)
# =============================================================================
def dumpData(data, name_str): #name_str = (str) name of pickle file, ex: name_str='indicators_normT1'
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
def reformat(indicators, y):
    indicators_new  = pd.DataFrame()
    y_new           = pd.DataFrame()
    for ticker in y:
        indicators_new = indicators_new.append(indicators[ticker])
        y_new          = y_new.append(y[ticker])
    return indicators_new, y_new
    