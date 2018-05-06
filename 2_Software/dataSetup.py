# =============================================================================
# Packages:
import quandl
import pandas as pd
import numpy as np
import talib as ta
import pickle
import copy
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
#    start_date = '2017-01-01' :More test data, never train
#    end_date = '2018-03-28'
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
    tickers_realestate = ['ECL', 'FMC', 'IP', 'VMC', 'BMS']
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
        
    #labels
    y = {}
    for ticker in data:
        y[ticker] = pd.DataFrame(np.concatenate((np.array([0.0]),np.sign(np.diff(data[ticker].iloc[:,3])))))
    
    return data, y
# =============================================================================
def genTA(data, y, t): #t is timeperiod
    indicators  = {}
    y_ind = copy.deepcopy(y)
   
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
        indicators[ticker]['ROC'] = ta.ROC(data[ticker].iloc[:,3], timeperiod=(t-1))
        indicators[ticker]['ROCP']= ta.ROCP(data[ticker].iloc[:,3],timeperiod=(t-1))
        indicators[ticker]['STOCH_SLOWK'], indicators[ticker]['STOCH_SLOWD'] = ta.STOCH(data[ticker].iloc[:,1], data[ticker].iloc[:,2], data[ticker].iloc[:,3], fastk_period=t, slowk_period=int(.6*t), slowk_matype=0, slowd_period=int(.6*t), slowd_matype=0)
        indicators[ticker]['MACD'], indicators[ticker]['MACDSIGNAL'], indicators[ticker]['MACDHIST'] = ta.MACD(data[ticker].iloc[:,3], fastperiod=t,slowperiod=2*t,signalperiod=int(.7*t))
        
    ## Volume
        indicators[ticker]['OBV'] = ta.OBV(data[ticker].iloc[:,3], data[ticker].iloc[:,4])
        indicators[ticker]['AD'] = ta.AD(data[ticker].iloc[:,1], data[ticker].iloc[:,2], data[ticker].iloc[:,3], data[ticker].iloc[:,4])
        indicators[ticker]['ADOSC'] = ta.ADOSC(data[ticker].iloc[:,1], data[ticker].iloc[:,2], data[ticker].iloc[:,3], data[ticker].iloc[:,4], fastperiod=int(.3*t), slowperiod=t)
        
    ## Cycle
        indicators[ticker]['HT_DCPERIOD'] = ta.HT_DCPERIOD(data[ticker].iloc[:,3])
        indicators[ticker]['HT_TRENDMODE']= ta.HT_TRENDMODE(data[ticker].iloc[:,3])
    
    ## Price
        indicators[ticker]['AVGPRICE'] = ta.AVGPRICE(data[ticker].iloc[:,0], data[ticker].iloc[:,1], data[ticker].iloc[:,2], data[ticker].iloc[:,3])
        indicators[ticker]['TYPPRICE'] = ta.TYPPRICE(data[ticker].iloc[:,1], data[ticker].iloc[:,2], data[ticker].iloc[:,3])
    
    ## Volatility
        indicators[ticker]['ATR'] = ta.ATR(data[ticker].iloc[:,1], data[ticker].iloc[:,2], data[ticker].iloc[:,3],  timeperiod=(t-1))
    
    ## Statistics
        indicators[ticker]['BETA'] = ta.BETA(data[ticker].iloc[:,1], data[ticker].iloc[:,2], timeperiod=(t-1))
        indicators[ticker]['LINEARREG'] = ta.LINEARREG(data[ticker].iloc[:,3], timeperiod=t)
        indicators[ticker]['VAR'] = ta.VAR(data[ticker].iloc[:,3], timeperiod=t, nbdev=1)
    
    ## Math Transform
        indicators[ticker]['EXP'] = ta.EXP(data[ticker].iloc[:,3])
        indicators[ticker]['LN'] = ta.LN(data[ticker].iloc[:,3])
    
    ## Patterns (returns integers - but norming might not really do anything but wondering if they should be normed)
        indicators[ticker]['CDLENGULFING'] = ta.CDLENGULFING(data[ticker].iloc[:,0], data[ticker].iloc[:,1], data[ticker].iloc[:,2], data[ticker].iloc[:,3])
        indicators[ticker]['CDLDOJI']      = ta.CDLDOJI(data[ticker].iloc[:,0], data[ticker].iloc[:,1], data[ticker].iloc[:,2], data[ticker].iloc[:,3])
        indicators[ticker]['CDLHAMMER']    = ta.CDLHAMMER(data[ticker].iloc[:,0], data[ticker].iloc[:,1], data[ticker].iloc[:,2], data[ticker].iloc[:,3])
        indicators[ticker]['CDLHANGINGMAN']= ta.CDLHANGINGMAN(data[ticker].iloc[:,0], data[ticker].iloc[:,1], data[ticker].iloc[:,2], data[ticker].iloc[:,3])
        
    #drop 'nan' values
        indicators[ticker].drop(indicators[ticker].index[np.arange(0,63)], inplace=True)
        y_ind[ticker].drop(y_ind[ticker].index[np.arange(0,63)], inplace=True)
        
    #Normalize Features
    indicators_norm = normData(indicators)
        
    return indicators_norm, indicators, y_ind

# =============================================================================
def loadQdata():
    data = pickle.load(open('data/data.pickle', 'rb'))
    y    = pickle.load(open('data/y.pickle', 'rb'))
    return data, y
# =============================================================================
def loadTAdata(tNum): # tNum = 1 or 2 or 3 (int)
    if(tNum != 1 and tNum != 2 and tNum != 3):
        print('loadData:tNum must be integer 1, 2, or 3')
        exit()
        
    tNum_str = str(tNum)
        
    indicators_norm   = pickle.load(open('data/indicators_normT'+tNum_str+'.pickle', 'rb'))
    indicators        = pickle.load(open('data/indicatorsT'+tNum_str+'.pickle', 'rb'))
    y_ind             = pickle.load(open('data/y_indT'+tNum_str+'.pickle', 'rb'))
    
    featureNames = {'Indicators':list(indicators[list(indicators.keys())[0]].columns.values)}
    pd.DataFrame.from_dict(featureNames).to_csv('../3_Deliverables/Final Paper/data/features.csv', index=False)
    
    return(indicators_norm, indicators, y_ind)
# =============================================================================
def dumpData(data, name_str): #name_str = (str) name of pickle file, ex: name_str='indicators_normT1'
    pickle.dump(data,  open('data/'+name_str+'.pickle',  'wb'), protocol=pickle.HIGHEST_PROTOCOL)
# =============================================================================
def normData(dataIn):
    dataOut = copy.deepcopy(dataIn)
    for ticker in dataIn:
        columnNames = list(dataOut[ticker].columns.values)
        indexNames = list(dataOut[ticker].index.values)
        tempMin = np.min(dataIn[ticker], axis=0)
        tempMax = np.max(dataIn[ticker], axis=0)
        np.seterr(all='warn', divide='ignore', invalid='ignore')
        temp = np.divide(np.array((dataIn[ticker] - tempMin)),np.array((tempMax - tempMin))) 
        dataOut[ticker] = pd.DataFrame(np.nan_to_num(temp), index=indexNames, columns=columnNames)
    return dataOut
# =============================================================================
def reformat(x, y):
    x_new = pd.DataFrame()
    y_new = pd.DataFrame()
    for ticker in y:
        x_new = x_new.append(x[ticker])
        y_new = y_new.append(y[ticker])
    return x_new, y_new
    