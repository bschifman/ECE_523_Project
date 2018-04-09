import quandl
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
quandl.ApiConfig.api_key = 'HwQoB4ePcDi8bFzJ6SJA'
#%%
#start_date = '2006-12-31'
#end_date = '20017-12-31'
#database_code = 'WIKI'
#
##tickers_financials = []
##tickers_utilities = []
##tickers_energy = []
##tickers_healthcare = []
##tickers_technology = []
##tickers_realestate = []
#
##THIS TAKES ABOUT A MINUTE WITH THE CURRENT DATES
#tickers_financials = ['JPM', 'BAC', 'WFC', 'C', 'MS']
#tickers_utilities = ['T', 'VZ', 'NEE', 'TMUS']
#tickers_energy = ['XOM', 'CVX', 'BP', 'GE', 'SLB']
#tickers_healthcare = ['JNJ', 'UNH', 'PFE', 'MRK', 'ABBV', 'MMM', 'AMGN', 'MDT']
#tickers_technology = ['AAPL', 'GOOGL', 'MSFT', 'FB', 'INTC', 'CSCO', 'ORCL', 'IBM', 'NVDA']
#tickers_realestate = ['ECL', 'DWDP', 'FMC', 'IP', 'PPG', 'VMC', 'BMS']
#tickers = [tickers_financials, tickers_utilities, tickers_energy, tickers_healthcare, tickers_technology, tickers_realestate]
#    
#data = {}
#
##data = quandl.get('WIKI/' + tickers, start_date=start_date, end_date=end_date)
#
#def import_data(ticker, start_date, end_date):
#    return quandl.get('WIKI/' + ticker, start_date=start_date, end_date=end_date)
#
#for index in tickers:
#    for ticker in index:
#        print(ticker)
#        data[ticker] = import_data(ticker, start_date, end_date)
#        
#with open('data.pickle', 'wb') as handle:
#    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
#%%
with open('data.pickle', 'rb') as handle:
    data = pickle.load(handle)

data_sma20 = {}
data_bollinger_low = {}
data_bollinger_high = {}

test = data['JPM'].iloc[:, 0:4]

for ticker in data:
    data_sma20[ticker] = data[ticker].iloc[:, 0:4].rolling(window=20).mean()
    data_sma20[ticker]['Open_std'] = data[ticker].iloc[:,0].rolling(window=20).std()
    data_sma20[ticker]['High_std'] = data[ticker].iloc[:,1].rolling(window=20).std() 
    data_sma20[ticker]['Low_std'] = data[ticker].iloc[:,2].rolling(window=20).std() 
    data_sma20[ticker]['Close_std'] = data[ticker].iloc[:,3].rolling(window=20).std() 
    data_bollinger_low[ticker] = data_sma20[ticker]['Open'] - data_sma20[ticker]['Open_std']*2
    data_bollinger_high[ticker] = data_sma20[ticker]['Open'] + data_sma20[ticker]['Open_std']*2

plt.plot(data['JPM']['Open'])
plt.plot(data_sma20['JPM']['Open'])
plt.plot(data_bollinger_low['JPM'])
plt.plot(data_bollinger_high['JPM'])

