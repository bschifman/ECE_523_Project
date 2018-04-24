import numpy as np
import pandas as pd
#import Parser

#%% Daily Labels
daily = {}
for ticker in data:
    daily[ticker]           = pd.DataFrame(np.sign(np.vstack(([0],np.diff(data[ticker].iloc[:,0])))))
    daily[ticker].rename(columns={daily[ticker ].columns[0]: "Open"}, inplace=True)
    daily[ticker]['High']   = pd.DataFrame(np.sign(np.diff(data[ticker].iloc[:,1])))
    daily[ticker]['Low']    = pd.DataFrame(np.sign(np.diff(data[ticker].iloc[:,2])))
    daily[ticker]['Close']  = pd.DataFrame(np.sign(np.diff(data[ticker].iloc[:,3])))
    daily[ticker]['Volume'] = pd.DataFrame(np.sign(np.diff(data[ticker].iloc[:,4])))
    

#%% Feature Selection