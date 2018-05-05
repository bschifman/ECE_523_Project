# -*- coding: utf-8 -*-
"""
Created on Fri May  4 21:21:12 2018

@author: bschifman
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt

input = (data['JPM'].iloc[:,0]).as_matrix()
std = temp.std()
temp = pywt.dwt(input, 'haar')
temp = pywt.threshold(temp, 3, mode='hard')
output = pywt.idwt(temp[0,:], temp[1,:], 'haar')

plt.figure()
plt.plot(input, label='INPUT')
plt.plot(output, label='HWT')
plt.legend()