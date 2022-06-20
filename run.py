import os

import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)

import sys

import warnings                                  # `do not disturbe` mode
warnings.filterwarnings('ignore')


sys.path.insert(0, 'lib/')
#sys.tracebacklimit = 0 # Hide traceback on errors

######
from file_selector import file_selector
from sidebar_menus import sidebar_menus
from holtwinter import holt_winter_st
from SARIMA import SARIMA_st
from ML_models import ML_models_st
from LSTM import LSTM_st

from run_all import run_all_st
#from timelit import run_all
######

#import timelit

import numpy as np                               # vectors and matrices
import pandas as pd                              # tables and data manipulations

import matplotlib.pyplot as plt                  # plots
from matplotlib.pyplot import show, draw, ion
#%matplotlib inline

import seaborn as sns                            # more plots

from dateutil.relativedelta import relativedelta # working with dates with style
import time
from time import sleep
from stqdm import stqdm

from scipy.optimize import minimize              # for function minimization

import statsmodels.formula.api as smf            # statistics and econometrics
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs

from itertools import product                    # some useful functions
from tqdm import tqdm_notebook

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

from numpy import log

from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import TimeSeriesSplit # you have everything done for you
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

import pylab

from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense, LSTM

from xgboost import XGBRegressor 


## Evaluation Metrics
def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def moving_average(series, n):
    """
        Calculate average of last n observations
    """
    return np.average(series[-n:])

#############################

# Defining Containers
description = st.container()
sidebar = st.container()
data_preview = st.container()
forecasting = st.container()

############################


### DESCRIPTION

with description:
    pd.set_option('display.float_format', lambda x: '%.3f' % x) # Granting that pandas won't use scientific notation for floating fields

    description =   '''
                    **TimeLit** is a one-stop time series forecasting solution. It will help you to forecast the future data
                    from historical uni-variate time series. It uses multiple statistical, Machine Learning and LSTM models
                    on your provided dataset to find the most accurate and optimal model for your specific time series dataset 
                    and predicts future data with that model.
                    '''
    # Description
    st.image('img/banner.png')
    st.write('*Your one stop time-series forecasting solution*')
    st.write(description)


### SIDEBAR AND DATA PREPROCESSING

with sidebar:
    st.sidebar.title('Your data')

    filename, dfx = file_selector()
    
    st.sidebar.markdown('### Choosing columns')
    pred = st.sidebar.selectbox('Which column do you want to PREDICT?', dfx.columns, 1)
    ds = st.sidebar.selectbox('Which column is your DATE column?', dfx.columns, 0)
    
    cols= [ds, pred]
    df = dfx[cols].copy()
    df[ds] = pd.to_datetime(df[ds])
    df.set_index(ds, inplace=True)
    
    n_len = len(df.index)
    
    
### Data Preview
with data_preview:    
    
    st.markdown('### **Your time-series plot**')
    plt.figure(figsize=(15, 5))
    plt.plot(df[pred])
    plt.title('%s'%pred)
    plt.grid(True)
    st.pyplot()
    
    st.markdown('### **First rows of your data**')
    st.dataframe(df.head(10)) # First lines of DataFrame

    st.sidebar.markdown('Number of instances in your dataset:')
    st.sidebar.markdown(n_len)
    
    
    #Plot TS
    
    # Downsample
    dw = st.selectbox('Do you want to resample your data', ['No','Yes'])
    
    if dw == 'y' or dw =='Y' or dw =='yes' or dw =='Yes' or dw =='YES':
        
        rs = st.selectbox('To which order do you want to resample?:', ['H','D','W', 'M','Y'])
        agg = st.selectbox('How do you want to aggregate your predictor?:', ['mean','sum'])
        
        if agg == 'sum':
            df = df.resample(rs).sum()

        if agg == 'mean':
            df = df.resample(rs).mean()
        
        n_len2 = len(df.index)
        st.sidebar.markdown('Number of instances after resampling:')
        st.sidebar.markdown(n_len2)
         
        st.markdown('## **Resampled dataset preview**')
        plt.plot(df[pred])
        #plt.figure(figsize=(15,5))
        plt.title('%s'%pred)
        plt.grid(True)
        st.pyplot()
        
        st.dataframe(df.head(10)) # Resampled df
        
    else:
        pass
    

### Forecasting

with forecasting:
    
    st.markdown('## **Now lets forecast!**')
    
    season = int(st.number_input('''How many observations is one season?
                                    (for example: hourly data would usually have 24 as season size
                                    monthly data would usually have 12 as season size etc.)'''
                                    ,min_value = 1.0, step=1.0))
    
    fc = st.selectbox('Select the model you want to run:'
                      , ['Select a model'
                        , 'Holt-Winters Method (Triple Exponential Smoothening)'
                        , 'SARIMA'
                        , 'Machine Learning Models'
                        , 'LSTM'
                        #, 'Run All (without LSTM)'
                        , 'Run All Models'
                        ])
    
    
    if fc == 'Select a model':
        pass
    
    holtwinter_con = st.container()
    with holtwinter_con:
        if fc == 'Holt-Winters Method (Triple Exponential Smoothening)':
            hw_acc = holt_winter_st(df, season, pred, ds)
    
    
    SARIMA_con=st.container()    
    with SARIMA_con:    
        if fc == 'SARIMA':
            sarima_acc = SARIMA_st(df, season, pred, ds)
            
            
    ML_con=st.container()    
    with ML_con:    
        if fc == 'Machine Learning Models':
            ML_acc = ML_models_st(df, season, pred, ds)
            
            
    LSTM_con=st.container()    
    with LSTM_con:    
        if fc == 'LSTM':
            LSTM_acc = LSTM_st(df, season, pred, ds) 

    run_all_con = st.container()
    
    def run_all():
        if fc == 'Run All Models':
            hw_acc = holt_winter_st(df, season, pred, ds)
            sarima_acc = SARIMA_st(df, season, pred, ds)
            ML_acc = ML_models_st(df, season, pred, ds)
            LSTM_acc = LSTM_st(df, season, pred, ds)
            
            return hw_acc, sarima_acc, ML_acc, LSTM_acc
            
           
    with run_all_con:
        if fc == 'Run All Models':
            st.sidebar.markdown('### **Mean Absolute Percentage Error**')
            st.markdown('### **ENSURE THAT YOU RUN ALL THE MODELS AND IN THE SET ORDER**')
            acc_df = run_all_st(df, season, pred, ds)
            #acc_df.sort_values(by=['MAPE%'], ascending = False, inplace=True)

  

         