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

def SARIMA_st(df, season, pred, ds):
    
    df2 = df
    pred = pred
    ds = ds
    season = season
    
    st.markdown('### **Seasonal Autoregressive Integrated Moving Average (SARIMA) Model**')
    
    def tsplot(y, lags=None, figsize=(12, 7), style='bmh'):
        """
            Plot time series, its ACF and PACF, calculate Dickey–Fuller test

            y - timeseries
            lags - how many lags to include in ACF, PACF calculation
        """
        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        with plt.style.context(style):    
            fig = plt.figure(figsize=figsize)
            layout = (2, 2)
            ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
            acf_ax = plt.subplot2grid(layout, (1, 0))
            pacf_ax = plt.subplot2grid(layout, (1, 1))

            y.plot(ax=ts_ax)
            p_value = sm.tsa.stattools.adfuller(y)[1]
            ts_ax.set_title('Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}'.format(p_value))
            smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
            smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
            plt.tight_layout()
            plt.show()
            st.pyplot()

    ####################

    #Taking care of seasonality and stationary


    count_d = 0
    count_s = 0

    df_diff = df2[pred].copy()
    p_val = adfuller(df_diff.dropna())[1] #p value

    while p_val >= 0.05:
        print ('differencing for stationary.....')
        df_diff = df_diff - df_diff.shift(1)
        count_d +=1
        p_val = adfuller(df_diff.dropna())[1]

        if p_val<0.05:
            break

    tsplot(df_diff[season+1:], lags=(season*3))


    sz = st.radio('Observe the above plot; does seasonality still exist in your timeseries?'
                 ,('Select an option','Yes','No'))

    
    if sz == 'y' or sz =='Y' or sz =='yes' or sz =='Yes' or sz =='YES':
        st.markdown('differencing for seasonality.....')
        df_diff = df_diff - df_diff.shift(season)
        count_s += 1
        tsplot(df_diff[season+1:], lags=(season*3))

        more = st.radio('Do you want to shift by one more lag to remove seasonality?'
                 ,('Select an option','Yes','No'))
        
        if more == 'y' or more == 'yes' or more == 'Yes' or more == 'YES' or more == 'Y':
            df_diff = df_diff - df_diff.shift(1)
            count_s += 1
            tsplot(df_diff[season+1:], lags=(season*3))
            more = st.radio('Do you want to shift by one more lag to remove seasonality?'
                 ,('Select an option','Yes','No'), key='more2')
            
        if more == 'No':
            #print ('p values after adjusting for seasonality and or stationary: ', adf2)
            st.write ('##### Differenced for stationary:', count_d, 'times.')
            st.write ('##### Differenced for seasonality:', count_s, 'times.')


            if count_s > 0:
                D1=1 #For SARIMA Parameter
            else:
                D1=0

            if count_d > 0:
                d1=1 #For SARIMA parameter   
            else:
                d1=0


            ###########


            #print ("- $p$ is most probably 4 since it is the last significant lag on the PACF, after which, most others are not significant.")
            #print ("- $d$ equals 1 because we had first differences")



            st.markdown('''

              #### ***Ｄｅｔｅｒｍｉｎｉｎｇ　ｐａｒａｍｅｔｅｒ　ｖａｌｕｅｓ:***

              * 𝑝 : To determine the initial 𝑝, you need to look at the PACF plot above and find the biggest 
                  significant lag after which most other lags become insignificant.

              * 𝑞 : The initial value of 𝑞 can be found on the ACF plot above with the same logic as for 𝑝.

              * 𝑑 : The order of integration. This is simply the number of nonseasonal differences needed 
                  to make the series stationary. In our case, our model has auto differenced based on ADF
                  and has automatically taken a value of 𝑑.

              * 𝑠 : This is responsible for seasonality and equals the season period length of the series.
                  We have already input this in the start.

              * 𝑃 : order of autoregression for the seasonal component of the model, which can be derived from PACF. 
                  But you need to look at the number of significant lags, which are the multiples of the season 
                  period length. For example, if the period equals 24 and we see the 24-th and 48-th lags are 
                  significant in the PACF, that means the initial 𝑃  should be 2.

              * 𝑄 : Similar logic as 𝑃 but using the ACF plot instead.

              * 𝐷 : The order of seasonal integration. This can be equal to 1 or 0, depending on whether seasonal 
                  differeces were applied or not. Our model has automatically taken this value based on whether
                  we differenced or not.

              𝑮𝒊𝒗𝒆𝒏 𝒕𝒉𝒆 𝒂𝒃𝒐𝒗𝒆 𝒎𝒆𝒕𝒉𝒐𝒅, 𝒊𝒏𝒑𝒖𝒕 𝒂𝒏 𝒆𝒔𝒕𝒊𝒎𝒂𝒕𝒆𝒅 𝒓𝒂𝒏𝒈𝒆 𝒇𝒐𝒓 𝒕𝒉𝒆 𝒂𝒃𝒐𝒗𝒆 𝒗𝒂𝒍𝒖𝒆𝒔. 𝑩𝒂𝒔𝒆𝒅 𝒐𝒏 𝒑𝒓𝒐𝒗𝒊𝒅𝒆𝒅 𝒓𝒂𝒏𝒈𝒆𝒔 𝒐𝒖𝒓 𝒎𝒐𝒅𝒆𝒍 𝒘𝒊𝒍𝒍 
              𝒂𝒖𝒕𝒐-𝒅𝒆𝒕𝒆𝒓𝒎𝒊𝒏𝒆 𝒕𝒉𝒆 𝒃𝒆𝒔𝒕 𝒑𝒂𝒓𝒂𝒎𝒆𝒕𝒆𝒓𝒔 𝒇𝒐𝒓 𝒇𝒐𝒓𝒆𝒄𝒂𝒔𝒕𝒊𝒏𝒈. 

              **(𝑩𝒆𝒘𝒂𝒓𝒆!!: 𝒕𝒉𝒆 𝒍𝒂𝒓𝒈𝒆𝒓 𝒕𝒉𝒆 𝒓𝒂𝒏𝒈𝒆𝒔, 𝒕𝒉𝒆 𝒉𝒊𝒈𝒉𝒆𝒓 𝒕𝒉𝒆 𝒓𝒖𝒏𝒕𝒊𝒎𝒆 𝒇𝒐𝒓 𝒇𝒊𝒏𝒅𝒊𝒏𝒈 𝒕𝒉𝒆 𝒃𝒆𝒔𝒕 𝒑𝒂𝒓𝒂𝒎𝒆𝒕𝒆𝒓𝒔 𝒂𝒔 𝒊𝒕 𝒘𝒊𝒍𝒍 𝒓𝒖𝒏 𝒂𝒏𝒅 𝒕𝒆𝒔𝒕 𝒂𝒍𝒍 
              𝒄𝒐𝒎𝒃𝒊𝒏𝒂𝒕𝒊𝒐𝒏𝒔.)**

              ''')


            p1 = int(st.number_input('Starting range for 𝑝?'
                                        ,min_value = 0.0, step=1.0))
            p2 = int(st.number_input('Ending range for 𝑝?'
                                        ,min_value = 0.0, step=1.0, value=2.0))
            #d1 = int(input('input d = 0 if series has not been differenced, otherwise 1:  '))
            q1 = int(st.number_input('Starting range for 𝑞?'
                                        ,min_value = 0.0, step=1.0))
            q2 = int(st.number_input('Ending range for 𝑞?'
                                        ,min_value = 0.0, step=1.0, value=2.0))
            st.write('𝑑 = ',d1,'(automatically determined based on our differencing for stationary)')
            st.write('𝑠 = ',season,' (as per our input)')

            P1 = int(st.number_input('Starting range for 𝑃?'
                                        ,min_value = 0.0, step=1.0))
            P2 = int(st.number_input('Ending range for 𝑃?'
                                        ,min_value = 0.0, step=1.0, value=2.0))      
            #D1 = int(input('input D = 0 if series has not been differenced, otherwise 1:  '))
            Q1 = int(st.number_input('Starting range for 𝑄?'
                                        ,min_value = 0.0, step=1.0))
            Q2 = int(st.number_input('Ending range for 𝑄?'
                                        ,min_value = 0.0, step=1.0, value=2.0))

            st.write('𝐷 = ',D1,' (automatically determined based on our differencing for seasonality)')

            if st.button('Auto Tune Parameters & Forecast with SARIMA'):
                with st.spinner('Tuning parameters and forecasting with SARIMA...'):       
                    # setting initial values and some bounds for them
                    ps = range(p1, p2)
                    d=d1 
                    qs = range(q1, q2)
                    Ps = range(P1, P2)
                    D=D1 
                    Qs = range(Q1, Q2)
                    s = season # season length is still 24


                    # creating list with all the possible combinations of parameters
                    parameters = product(ps, qs, Ps, Qs)
                    parameters_list = list(parameters)
                    #len(parameters_list)


                    def optimizeSARIMA(parameters_list, d, D, s, series):
                        """
                            Return dataframe with parameters and corresponding AIC

                            parameters_list - list with (p, q, P, Q) tuples
                            d - integration order in ARIMA model
                            D - seasonal integration order 
                            s - length of season
                        """

                        results = []
                        best_aic = float("inf")

                        for param in tqdm_notebook(parameters_list):
                            # we need try-except because on some combinations model fails to converge
                            try:
                                model=sm.tsa.statespace.SARIMAX(series, order=(param[0], d, param[1]), 
                                                                seasonal_order=(param[2], D, param[3], s)).fit(disp=-1)
                            except:
                                continue
                            aic = model.aic
                            # saving best model, AIC and parameters
                            if aic < best_aic:
                                best_model = model
                                best_aic = aic
                                best_param = param
                            results.append([param, model.aic])

                        result_table = pd.DataFrame(results)
                        result_table.columns = ['parameters', 'aic']
                        # sorting in ascending order, the lower AIC is - the better
                        result_table = result_table.sort_values(by='aic', ascending=True).reset_index(drop=True)

                        return result_table

                    #if st.button('Auto Tune Parameters & Forecast with SARIMA'):
                    #print('Optimising SARIMA for best parameters....')
                    result_table = optimizeSARIMA(parameters_list, d, D, s, df2[pred])

                    len_res= len(result_table)
                    # set the parameters that give the lowest AIC
                    p, q, P, Q = result_table.parameters[0]


                    best_model=sm.tsa.statespace.SARIMAX(df2[pred], order=(p, d, q), 
                                                    seasonal_order=(P, D, Q, s)).fit(disp=-1)


                    st.write('##### MODEL SUMMARY')
                    st.write(best_model.summary())


                    def plotSARIMA(series, model, n_steps):
                        """
                            Plots model vs predicted values

                            series - dataset with timeseries
                            model - fitted SARIMA model
                            n_steps - number of steps to predict in the future

                        """
                        # adding model values
                        data = series.copy()
                        data.columns = ['actual']
                        data['arima_model'] = model.fittedvalues
                        # making a shift on s+d steps, because these values were unobserved by the model
                        # due to the differentiating
                        data['arima_model'][:s+d] = np.NaN

                        # forecasting on n_steps forward 
                        global forecast
                        forecast = model.predict(start = data.shape[0], end = data.shape[0]+n_steps)
                        forecast = data.arima_model.append(forecast)
                        forecast = pd.DataFrame (forecast)
                        # calculate error, again having shifted on s+d steps from the beginning
                        error = mean_absolute_percentage_error(data['actual'][s+d:], data['arima_model'][s+d:])

                        plt.figure(figsize=(15, 5))
                        plt.title("Mean Absolute Percentage Error: {0:.2f}%".format(error))
                        plt.plot(forecast, color='r', label="model")
                        plt.axvspan(data.index[-1], forecast.index[-1], alpha=0.5, color='lightgrey')
                        plt.plot(data.actual, label="actual")
                        plt.legend()
                        plt.grid(True)
                        plt.show()
                        st.pyplot()

                        global mape
                        mape = []
                        mape.append(['SARIMA', error])

                        global mape_df
                        mape_df = pd.DataFrame(data=mape, columns=['Model', 'MAPE%'])
                        mape_df.drop_duplicates(inplace=True)

                    plotSARIMA(df2, best_model, 50)

                    st.markdown('##### **Mean Absolute Percentage Error**')
                    st.dataframe(mape_df)
                    return mape_df 
            


    if sz == 'n' or sz =='no' or sz =='N' or sz =='NO' or sz =='No':
        st.markdown('Not differencing or lagging the timeseries further')



        #print ('p values after adjusting for seasonality and or stationary: ', adf2)
        st.write ('##### Differenced for stationary:', count_d, 'times.')
        st.write ('##### Differenced for seasonality:', count_s, 'times.')


        if count_s > 0:
            D1=1 #For SARIMA Parameter
        else:
            D1=0

        if count_d > 0:
            d1=1 #For SARIMA parameter   
        else:
            d1=0


        ###########


        #print ("- $p$ is most probably 4 since it is the last significant lag on the PACF, after which, most others are not significant.")
        #print ("- $d$ equals 1 because we had first differences")



        st.markdown('''

          #### ***Ｄｅｔｅｒｍｉｎｉｎｇ　ｐａｒａｍｅｔｅｒ　ｖａｌｕｅｓ:***

          * 𝑝 : To determine the initial 𝑝, you need to look at the PACF plot above and find the biggest 
              significant lag after which most other lags become insignificant.

          * 𝑞 : The initial value of 𝑞 can be found on the ACF plot above with the same logic as for 𝑝.

          * 𝑑 : The order of integration. This is simply the number of nonseasonal differences needed 
              to make the series stationary. In our case, our model has auto differenced based on ADF
              and has automatically taken a value of 𝑑.

          * 𝑠 : This is responsible for seasonality and equals the season period length of the series.
              We have already input this in the start.

          * 𝑃 : order of autoregression for the seasonal component of the model, which can be derived from PACF. 
              But you need to look at the number of significant lags, which are the multiples of the season 
              period length. For example, if the period equals 24 and we see the 24-th and 48-th lags are 
              significant in the PACF, that means the initial 𝑃  should be 2.

          * 𝑄 : Similar logic as 𝑃 but using the ACF plot instead.

          * 𝐷 : The order of seasonal integration. This can be equal to 1 or 0, depending on whether seasonal 
              differeces were applied or not. Our model has automatically taken this value based on whether
              we differenced or not.

          𝑮𝒊𝒗𝒆𝒏 𝒕𝒉𝒆 𝒂𝒃𝒐𝒗𝒆 𝒎𝒆𝒕𝒉𝒐𝒅, 𝒊𝒏𝒑𝒖𝒕 𝒂𝒏 𝒆𝒔𝒕𝒊𝒎𝒂𝒕𝒆𝒅 𝒓𝒂𝒏𝒈𝒆 𝒇𝒐𝒓 𝒕𝒉𝒆 𝒂𝒃𝒐𝒗𝒆 𝒗𝒂𝒍𝒖𝒆𝒔. 𝑩𝒂𝒔𝒆𝒅 𝒐𝒏 𝒑𝒓𝒐𝒗𝒊𝒅𝒆𝒅 𝒓𝒂𝒏𝒈𝒆𝒔 𝒐𝒖𝒓 𝒎𝒐𝒅𝒆𝒍 𝒘𝒊𝒍𝒍 
          𝒂𝒖𝒕𝒐-𝒅𝒆𝒕𝒆𝒓𝒎𝒊𝒏𝒆 𝒕𝒉𝒆 𝒃𝒆𝒔𝒕 𝒑𝒂𝒓𝒂𝒎𝒆𝒕𝒆𝒓𝒔 𝒇𝒐𝒓 𝒇𝒐𝒓𝒆𝒄𝒂𝒔𝒕𝒊𝒏𝒈. 

          **(𝑩𝒆𝒘𝒂𝒓𝒆!!: 𝒕𝒉𝒆 𝒍𝒂𝒓𝒈𝒆𝒓 𝒕𝒉𝒆 𝒓𝒂𝒏𝒈𝒆𝒔, 𝒕𝒉𝒆 𝒉𝒊𝒈𝒉𝒆𝒓 𝒕𝒉𝒆 𝒓𝒖𝒏𝒕𝒊𝒎𝒆 𝒇𝒐𝒓 𝒇𝒊𝒏𝒅𝒊𝒏𝒈 𝒕𝒉𝒆 𝒃𝒆𝒔𝒕 𝒑𝒂𝒓𝒂𝒎𝒆𝒕𝒆𝒓𝒔 𝒂𝒔 𝒊𝒕 𝒘𝒊𝒍𝒍 𝒓𝒖𝒏 𝒂𝒏𝒅 𝒕𝒆𝒔𝒕 𝒂𝒍𝒍 
          𝒄𝒐𝒎𝒃𝒊𝒏𝒂𝒕𝒊𝒐𝒏𝒔.)**

          ''')


        p1 = int(st.number_input('Starting range for 𝑝?'
                                    ,min_value = 0.0, step=1.0))
        p2 = int(st.number_input('Ending range for 𝑝?'
                                    ,min_value = 0.0, step=1.0, value=2.0))
        #d1 = int(input('input d = 0 if series has not been differenced, otherwise 1:  '))
        q1 = int(st.number_input('Starting range for 𝑞?'
                                    ,min_value = 0.0, step=1.0))
        q2 = int(st.number_input('Ending range for 𝑞?'
                                    ,min_value = 0.0, step=1.0, value=2.0))
        st.write('𝑑 = ',d1,'(automatically determined based on our differencing for stationary)')
        st.write('𝑠 = ',season,' (as per our input)')

        P1 = int(st.number_input('Starting range for 𝑃?'
                                    ,min_value = 0.0, step=1.0))
        P2 = int(st.number_input('Ending range for 𝑃?'
                                    ,min_value = 0.0, step=1.0, value=2.0))      
        #D1 = int(input('input D = 0 if series has not been differenced, otherwise 1:  '))
        Q1 = int(st.number_input('Starting range for 𝑄?'
                                    ,min_value = 0.0, step=1.0))
        Q2 = int(st.number_input('Ending range for 𝑄?'
                                    ,min_value = 0.0, step=1.0, value=2.0))

        st.write('𝐷 = ',D1,' (automatically determined based on our differencing for seasonality)')

        
        if st.button('Auto Tune Parameters & Forecast with SARIMA'):
            with st.spinner('Tuning parameters and forecasting with SARIMA...'):
                # setting initial values and some bounds for them
                ps = range(p1, p2)
                d=d1 
                qs = range(q1, q2)
                Ps = range(P1, P2)
                D=D1 
                Qs = range(Q1, Q2)
                s = season # season length is still 24


                # creating list with all the possible combinations of parameters
                parameters = product(ps, qs, Ps, Qs)
                parameters_list = list(parameters)
                #len(parameters_list)


                def optimizeSARIMA(parameters_list, d, D, s, series):
                    """
                        Return dataframe with parameters and corresponding AIC

                        parameters_list - list with (p, q, P, Q) tuples
                        d - integration order in ARIMA model
                        D - seasonal integration order 
                        s - length of season
                    """

                    results = []
                    best_aic = float("inf")

                    for param in tqdm_notebook(parameters_list):
                        # we need try-except because on some combinations model fails to converge
                        try:
                            model=sm.tsa.statespace.SARIMAX(series, order=(param[0], d, param[1]), 
                                                            seasonal_order=(param[2], D, param[3], s)).fit(disp=-1)
                        except:
                            continue
                        aic = model.aic
                        # saving best model, AIC and parameters
                        if aic < best_aic:
                            best_model = model
                            best_aic = aic
                            best_param = param
                        results.append([param, model.aic])

                    result_table = pd.DataFrame(results)
                    result_table.columns = ['parameters', 'aic']
                    # sorting in ascending order, the lower AIC is - the better
                    result_table = result_table.sort_values(by='aic', ascending=True).reset_index(drop=True)

                    return result_table

                #if st.button('Auto Tune Parameters & Forecast with SARIMA'):
                #print('Optimising SARIMA for best parameters....')
                result_table = optimizeSARIMA(parameters_list, d, D, s, df2[pred])

                len_res= len(result_table)
                # set the parameters that give the lowest AIC
                p, q, P, Q = result_table.parameters[0]


                best_model=sm.tsa.statespace.SARIMAX(df2[pred], order=(p, d, q), 
                                                seasonal_order=(P, D, Q, s)).fit(disp=-1)


                st.write('##### MODEL SUMMARY')
                st.write(best_model.summary())


                def plotSARIMA(series, model, n_steps):
                    """
                        Plots model vs predicted values

                        series - dataset with timeseries
                        model - fitted SARIMA model
                        n_steps - number of steps to predict in the future

                    """
                    # adding model values
                    data = series.copy()
                    data.columns = ['actual']
                    data['arima_model'] = model.fittedvalues
                    # making a shift on s+d steps, because these values were unobserved by the model
                    # due to the differentiating
                    data['arima_model'][:s+d] = np.NaN

                    # forecasting on n_steps forward 
                    global forecast
                    forecast = model.predict(start = data.shape[0], end = data.shape[0]+n_steps)
                    forecast = data.arima_model.append(forecast)
                    forecast = pd.DataFrame (forecast)
                    # calculate error, again having shifted on s+d steps from the beginning
                    error = mean_absolute_percentage_error(data['actual'][s+d:], data['arima_model'][s+d:])

                    plt.figure(figsize=(15, 5))
                    plt.title("Mean Absolute Percentage Error: {0:.2f}%".format(error))
                    plt.plot(forecast, color='r', label="model")
                    plt.axvspan(data.index[-1], forecast.index[-1], alpha=0.5, color='lightgrey')
                    plt.plot(data.actual, label="actual")
                    plt.legend()
                    plt.grid(True)
                    plt.show()
                    st.pyplot()

                    global mape
                    mape = []
                    mape.append(['SARIMA', error])

                    global mape_df
                    mape_df = pd.DataFrame(data=mape, columns=['Model', 'MAPE%'])
                    mape_df.drop_duplicates(inplace=True)

                plotSARIMA(df2, best_model, 50)

                st.markdown('##### **Mean Absolute Percentage Error**')
                st.dataframe(mape_df)
                return mape_df

