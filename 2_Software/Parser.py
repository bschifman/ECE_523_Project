import quandl
import pandas as pd
import numpy as np
import talib as ta
import pickle
import matplotlib.pyplot as plt
quandl.ApiConfig.api_key = 'HwQoB4ePcDi8bFzJ6SJA'
#%% Generate Pickle %##
# =============================================================================
# start_date = '2006-12-31'
# end_date = '2017-12-31'
# database_code = 'WIKI'
# 
# #tickers_financials = []
# #tickers_utilities = []
# #tickers_energy = []
# #tickers_healthcare = []
# #tickers_technology = []
# #tickers_realestate = []
# 
# #THIS TAKES ABOUT A MINUTE WITH THE CURRENT DATES
# tickers_financials = ['JPM', 'BAC', 'WFC', 'C', 'MS']
# tickers_utilities = ['T', 'VZ', 'NEE', 'TMUS']
# tickers_energy = ['XOM', 'CVX', 'BP', 'GE', 'SLB']
# tickers_healthcare = ['JNJ', 'UNH', 'PFE', 'MRK', 'ABBV', 'MMM', 'AMGN', 'MDT']
# tickers_technology = ['AAPL', 'GOOGL', 'MSFT', 'FB', 'INTC', 'CSCO', 'ORCL', 'IBM', 'NVDA']
# tickers_realestate = ['ECL', 'DWDP', 'FMC', 'IP', 'PPG', 'VMC', 'BMS']
# tickers = [tickers_financials, tickers_utilities, tickers_energy, tickers_healthcare, tickers_technology, tickers_realestate]
#     
# data = {}
# 
# #data = quandl.get('WIKI/' + tickers, start_date=start_date, end_date=end_date)
# 
# def import_data(ticker, start_date, end_date):
#     return quandl.get('WIKI/' + ticker, start_date=start_date, end_date=end_date)
# 
# for index in tickers:
#     for ticker in index:
#         print(ticker)
#         data[ticker] = import_data(ticker, start_date, end_date)
# #%%REFORMAT DATA 
# for ticker in data:
#     data[ticker].drop(['Open', 'High', 'Low', 'Close', 'Volume', 'Ex-Dividend', 'Split Ratio'], axis=1, inplace=True)
#     data[ticker].rename(index=str, columns={"Adj. Open": "Open", "Adj. High": "High", "Adj. Low": "Low", "Adj. Close":
#         "Close", "Adj. Volume": "Volume"}, inplace=True)
#     
#     pickle.dump(data, open('data/data.pickle', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
# =============================================================================
#%%
data = pickle.load(open('data/data.pickle', 'rb'))

data_new   = {}
data_open  = {}
data_high  = {}
data_low   = {}
data_close = {}

#labels
sign_daily = {}


# =============================================================================
#   #Data all in data_new
#   data_new[ticker]              = ta.SMA(data[ticker].iloc[:,0], timeperiod=20).to_frame()
#   data_new[ticker]['SMA_High' ] = ta.SMA(data[ticker].iloc[:,1], timeperiod=20)
#   data_new[ticker]['SMA_Low'  ] = ta.SMA(data[ticker].iloc[:,2], timeperiod=20)
#   data_new[ticker]['SMA_Close'] = ta.SMA(data[ticker].iloc[:,3], timeperiod=20)
#   data_new[ticker]['RSI_Open' ] = ta.RSI(data[ticker].iloc[:,0], timeperiod=14)
#   data_new[ticker]['RSI_High' ] = ta.RSI(data[ticker].iloc[:,1], timeperiod=14)
#   data_new[ticker]['RSI_Low'  ] = ta.RSI(data[ticker].iloc[:,2], timeperiod=14)
#   data_new[ticker]['RSI_Close'] = ta.RSI(data[ticker].iloc[:,3], timeperiod=14)
#   data_new[ticker]['OBV_Open' ] = ta.OBV(data[ticker].iloc[:,0], data[ticker].iloc[:,4])
#   data_new[ticker]['OBV_High' ] = ta.OBV(data[ticker].iloc[:,1], data[ticker].iloc[:,4])
#   data_new[ticker]['OBV_Low'  ] = ta.OBV(data[ticker].iloc[:,2], data[ticker].iloc[:,4])
#   data_new[ticker]['OBV_Close'] = ta.OBV(data[ticker].iloc[:,3], data[ticker].iloc[:,4])
#   data_open[ticker ]['EMA' ] = ta.EMA(data[ticker].iloc[:,0], timeperiod=20)
#   data_high[ticker ]['EMA' ] = ta.EMA(data[ticker].iloc[:,1], timeperiod=20)
#   data_low[ticker  ]['EMA' ] = ta.EMA(data[ticker].iloc[:,2], timeperiod=20)
#   data_close[ticker]['EMA' ] = ta.EMA(data[ticker].iloc[:,3], timeperiod=20)
#   data_new[ticker].rename(columns={data_new[ticker].columns[0]: "SMA_Open"}, inplace=True)
# =============================================================================

for ticker in data:
    #data[ticker]{'y'}  = (data[ticker].iloc[:,0])
    data_open[ticker ] = ta.SMA(data[ticker].iloc[:,0], timeperiod=20).to_frame()
    data_high[ticker ] = ta.SMA(data[ticker].iloc[:,1], timeperiod=20).to_frame()
    data_low[ticker  ] = ta.SMA(data[ticker].iloc[:,2], timeperiod=20).to_frame()
    data_close[ticker] = ta.SMA(data[ticker].iloc[:,3], timeperiod=20).to_frame()

    data_open[ticker ]['RSI' ] = ta.RSI(data[ticker].iloc[:,0], timeperiod=14)
    data_high[ticker ]['RSI' ] = ta.RSI(data[ticker].iloc[:,1], timeperiod=14)
    data_low[ticker  ]['RSI' ] = ta.RSI(data[ticker].iloc[:,2], timeperiod=14)
    data_close[ticker]['RSI' ] = ta.RSI(data[ticker].iloc[:,3], timeperiod=14)

    data_open[ticker ]['OBV'  ] = ta.OBV(data[ticker].iloc[:,0], data[ticker].iloc[:,4])
    data_high[ticker ]['OBV'  ] = ta.OBV(data[ticker].iloc[:,1], data[ticker].iloc[:,4])
    data_low[ticker  ]['OBV'  ] = ta.OBV(data[ticker].iloc[:,2], data[ticker].iloc[:,4])
    data_close[ticker]['OBV'  ] = ta.OBV(data[ticker].iloc[:,3], data[ticker].iloc[:,4])

    data_open[ticker ]['EMA' ] = ta.EMA(data[ticker].iloc[:,0], timeperiod=20)
    data_high[ticker ]['EMA' ] = ta.EMA(data[ticker].iloc[:,1], timeperiod=20)
    data_low[ticker  ]['EMA' ] = ta.EMA(data[ticker].iloc[:,2], timeperiod=20)
    data_close[ticker]['EMA' ] = ta.EMA(data[ticker].iloc[:,3], timeperiod=20)

    data_open[ticker ]['BBAND_Upper'], data_open[ticker ]['BBAND_Middle' ], data_open[ticker ]['BBAND_Lower' ] = ta.BBANDS(data[ticker].iloc[:,0], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    data_high[ticker ]['BBAND_Upper'], data_high[ticker ]['BBAND_Middle' ], data_high[ticker ]['BBAND_Lower' ] = ta.BBANDS(data[ticker].iloc[:,1], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    data_low[ticker  ]['BBAND_Upper'], data_low[ticker  ]['BBAND_Middle' ], data_low[ticker  ]['BBAND_Lower' ] = ta.BBANDS(data[ticker].iloc[:,2], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    data_close[ticker]['BBAND_Upper'], data_close[ticker]['BBAND_Middle' ], data_close[ticker]['BBAND_Lower' ] = ta.BBANDS(data[ticker].iloc[:,3], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)  

    data_open[ticker ]['ATR' ] = ta.ATR(data[ticker].iloc[:,1], data[ticker].iloc[:,2], data[ticker].iloc[:,3],  timeperiod=100)
    data_high[ticker ]['ATR' ] = ta.ATR(data[ticker].iloc[:,1], data[ticker].iloc[:,2], data[ticker].iloc[:,3],  timeperiod=100)
    data_low[ticker  ]['ATR' ] = ta.ATR(data[ticker].iloc[:,1], data[ticker].iloc[:,2], data[ticker].iloc[:,3],  timeperiod=100)
    data_close[ticker]['ATR' ] = ta.ATR(data[ticker].iloc[:,1], data[ticker].iloc[:,2], data[ticker].iloc[:,3],  timeperiod=100)
    
    data_open[ticker ]['MOM' ] = ta.MOM(data[ticker].iloc[:,0], timeperiod=10)
    data_high[ticker ]['MOM' ] = ta.MOM(data[ticker].iloc[:,1], timeperiod=10)
    data_low[ticker  ]['MOM' ] = ta.MOM(data[ticker].iloc[:,2], timeperiod=10)
    data_close[ticker]['MOM' ] = ta.MOM(data[ticker].iloc[:,3], timeperiod=10)
    
    data_open[ticker ].rename(columns={data_open[ticker ].columns[0]: "SMA"}, inplace=True)
    data_high[ticker ].rename(columns={data_high[ticker ].columns[0]: "SMA"}, inplace=True)
    data_low[ticker  ].rename(columns={data_low[ticker  ].columns[0]: "SMA"}, inplace=True)
    data_close[ticker].rename(columns={data_close[ticker].columns[0]: "SMA"}, inplace=True)

    #Daily Labels
    sign_daily[ticker]           = pd.DataFrame(np.sign(np.diff(data[ticker].iloc[:,0])))
    sign_daily[ticker].rename(columns={sign_daily[ticker ].columns[0]: "Open"}, inplace=True)
    sign_daily[ticker]['High']   = pd.DataFrame(np.sign(np.diff(data[ticker].iloc[:,1])))
    sign_daily[ticker]['Low']    = pd.DataFrame(np.sign(np.diff(data[ticker].iloc[:,2])))
    sign_daily[ticker]['Close']  = pd.DataFrame(np.sign(np.diff(data[ticker].iloc[:,3])))

#%%Dump Pickles
pickle.dump(data_open, open('data/data_open.pickle', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)  
pickle.dump(data_high, open('data/data_high.pickle', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)  
pickle.dump(data_low, open('data/data_low.pickle', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)  
pickle.dump(data_close, open('data/data_close.pickle', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

pickle.dump(sign_daily, open('data/sign_daily.pickle', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
