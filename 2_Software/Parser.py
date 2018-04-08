import quandl
import numpy as np
quandl.ApiConfig.api_key = 'HwQoB4ePcDi8bFzJ6SJA'
#%%
start_date = '2006-12-31'
end_date = '20017-12-31'
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

data = {}

#data = quandl.get('WIKI/' + tickers, start_date=start_date, end_date=end_date)

def import_data(ticker, start_date, end_date):
    return quandl.get('WIKI/' + ticker, start_date=start_date, end_date=end_date)

for index in tickers:
    for ticker in index:
        print(ticker)
        data[ticker] = import_data(ticker, start_date, end_date)
#%%