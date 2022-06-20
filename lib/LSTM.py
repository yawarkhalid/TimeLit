#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import sys

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


### ------------------------------------ ###


def LSTM_st(df, season, pred, ds):
    
    st.markdown('### **Long Short Term Model (LSTM)**')
    
    df=df
    pred = pred
    ds=ds
    season=season
    
    test_size = st.number_input('What is your test dataset size?'
                                ,min_value = 0.0, max_value=0.45, step=0.05, value=0.1)
    
    epochs = int(st.number_input('Over how many epochs do you want to train the LSTM model? (Higher the number of epochs, higher the time and compute power required to train)'
                                ,min_value = 1.0, step=1.0, value=1.0
                                ))
    
    if st.button('Train LSTM and Forecast'):
        with st.spinner('Training you LSTM Model...'):
            train = df.iloc[:(round((1-test_size)*len(df)))]
            test= df.iloc[(round((1-test_size)*len(df))):]

            test= test[0:(round((1-test_size)*len(df)))]    

            scaler = MinMaxScaler()
            scaler.fit(train)
            scaled_train = scaler.transform(train)
            scaled_test = scaler.transform(test) 

            from keras.preprocessing.sequence import TimeseriesGenerator

            from keras.models import Sequential
            from keras.layers import Dense, LSTM

            n_input = (round(((test_size)*len(df))))
            n_feature = 1

            train_generator = TimeseriesGenerator(scaled_train,scaled_train,length=n_input, batch_size=1)

            #if st.button('Train LSTM and Forecast'):
            ## Compiling LSTM
            model = Sequential()

            model.add(LSTM(128,activation = 'LeakyReLU', input_shape= (n_input, n_feature), return_sequences=True))
            model.add(LSTM(128, activation='LeakyReLU', return_sequences=True))
            model.add(LSTM(128, activation='LeakyReLU', return_sequences=False))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mse')

            model.summary()

            ## Fitting model

            model.fit_generator(train_generator,epochs= epochs)

            my_loss= model.history.history['loss']
            plt.plot(range(len(my_loss)),my_loss)
            plt.show()
            st.pyplot()

            first_eval_batch = scaled_train[-(round((test_size)*len(df))):]

            first_eval_batch = first_eval_batch.reshape((1,n_input,n_feature))

            #predicting
            model.predict(first_eval_batch)


            #holding my predictions
            test_predictions = []


            # last n_input points from the training set
            first_eval_batch = scaled_train[-n_input:]

            # reshape this to the format RNN wants (same format as TimeseriesGeneration)
            current_batch = first_eval_batch.reshape((1,n_input,n_feature))

            #how far into the future will I forecast?

            for i in range(len(test)):

                # One timestep ahead of historical n points
                current_pred = model.predict(current_batch)[0]

                #store that prediction
                test_predictions.append(current_pred)

                # UPDATE current batch o include prediction
                current_batch = np.append(current_batch[:,1:,:],[[current_pred]], axis= 1)

            true_predictions = scaler.inverse_transform(test_predictions)

            test['Predictions'] = true_predictions

            prediction=test['Predictions']
            true = test[pred]

            error = mean_absolute_percentage_error(prediction, true)

            def plot_lstm(error, test):

                test.plot(figsize=(15,7), title= ("Mean absolute percentage error {0:.2f}%".format(error)))
                plt.show()
                st.pyplot()

                global mape
                mape=[]
                mape.append(['LSTM', error])

                global mape_df
                mape_df = pd.DataFrame(data=mape, columns=['Model', 'MAPE%'])
                mape_df.drop_duplicates(inplace=True)

            plot_lstm(error,test)
            st.dataframe (mape_df)
            return mape_df