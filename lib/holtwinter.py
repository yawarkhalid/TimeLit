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
    return np.average(series[-n:])

#######------------------#########



def holt_winter_st(df, season, pred, ds):
    
    df1 = df
    pred = pred
    ds = ds
    season = season

    st.markdown('### **Holt-Winter Method (Triple Exponential Smoothing)**')

    if st.button('Forecast NOW with Holt-Winters'):
        with st.spinner('Forecasting with Holt-Winters...'):
            def plotMovingAverage(series, window, plot_intervals=False, scale=1.96, plot_anomalies=False):

                """
                    series - dataframe with timeseries
                    window - rolling window size 
                    plot_intervals - show confidence intervals
                    plot_anomalies - show anomalies 

                """
                rolling_mean = series.rolling(window=window).mean()

                plt.figure(figsize=(15,5))
                plt.title("Moving average\n window size = {}".format(window))
                plt.plot(rolling_mean, "g", label="Rolling mean trend")

                # Plot confidence intervals for smoothed values
                if plot_intervals:
                    mae = mean_absolute_error(series[window:], rolling_mean[window:])
                    deviation = np.std(series[window:] - rolling_mean[window:])
                    lower_bond = rolling_mean - (mae + scale * deviation)
                    upper_bond = rolling_mean + (mae + scale * deviation)
                    plt.plot(upper_bond, "r--", label="Upper Bond / Lower Bond")
                    plt.plot(lower_bond, "r--")

                    # Having the intervals, find abnormal values
                    if plot_anomalies:
                        anomalies = pd.DataFrame(index=series.index, columns=series.columns)
                        anomalies[series<lower_bond] = series[series<lower_bond]
                        anomalies[series>upper_bond] = series[series>upper_bond]
                        plt.plot(anomalies, "ro", markersize=10)

                plt.plot(series[window:], label="Actual values")
                plt.legend(loc="upper left")
                plt.grid(True)
                st.pyplot()

            #Plotting MA
            plotMovingAverage(df1, season)


            #HoltWinters
            class HoltWinters:

                """
                Holt-Winters model with the anomalies detection using Brutlag method

                # series - initial time series
                # slen - length of a season
                # alpha, beta, gamma - Holt-Winters model coefficients
                # n_preds - predictions horizon
                # scaling_factor - sets the width of the confidence interval by Brutlag (usually takes values from 2 to 3)

                """


                def __init__(self, series, slen, alpha, beta, gamma, n_preds, scaling_factor=1.96):
                    self.series = series
                    self.slen = slen
                    self.alpha = alpha
                    self.beta = beta
                    self.gamma = gamma
                    self.n_preds = n_preds
                    self.scaling_factor = scaling_factor


                def initial_trend(self):
                    sum = 0.0
                    for i in range(self.slen):
                        sum += float(self.series[i+self.slen] - self.series[i]) / self.slen
                    return sum / self.slen  

                def initial_seasonal_components(self):
                    seasonals = {}
                    season_averages = []
                    n_seasons = int(len(self.series)/self.slen)
                    # let's calculate season averages
                    for j in range(n_seasons):
                        season_averages.append(sum(self.series[self.slen*j:self.slen*j+self.slen])/float(self.slen))
                    # let's calculate initial values
                    for i in range(self.slen):
                        sum_of_vals_over_avg = 0.0
                        for j in range(n_seasons):
                            sum_of_vals_over_avg += self.series[self.slen*j+i]-season_averages[j]
                        seasonals[i] = sum_of_vals_over_avg/n_seasons
                    return seasonals   


                def triple_exponential_smoothing(self):
                    self.result = []
                    self.Smooth = []
                    self.Season = []
                    self.Trend = []
                    self.PredictedDeviation = []
                    self.UpperBond = []
                    self.LowerBond = []

                    seasonals = self.initial_seasonal_components()

                    for i in range(len(self.series)+self.n_preds):
                        if i == 0: # components initialization
                            smooth = self.series[0]
                            trend = self.initial_trend()
                            self.result.append(self.series[0])
                            self.Smooth.append(smooth)
                            self.Trend.append(trend)
                            self.Season.append(seasonals[i%self.slen])

                            self.PredictedDeviation.append(0)

                            self.UpperBond.append(self.result[0] + 
                                                  self.scaling_factor * 
                                                  self.PredictedDeviation[0])

                            self.LowerBond.append(self.result[0] - 
                                                  self.scaling_factor * 
                                                  self.PredictedDeviation[0])
                            continue

                        if i >= len(self.series): # predicting
                            m = i - len(self.series) + 1
                            self.result.append((smooth + m*trend) + seasonals[i%self.slen])

                            # when predicting we increase uncertainty on each step
                            self.PredictedDeviation.append(self.PredictedDeviation[-1]*1.01) 

                        else:
                            val = self.series[i]
                            last_smooth, smooth = smooth, self.alpha*(val-seasonals[i%self.slen]) + (1-self.alpha)*(smooth+trend)
                            trend = self.beta * (smooth-last_smooth) + (1-self.beta)*trend
                            seasonals[i%self.slen] = self.gamma*(val-smooth) + (1-self.gamma)*seasonals[i%self.slen]
                            self.result.append(smooth+trend+seasonals[i%self.slen])

                            # Deviation is calculated according to Brutlag algorithm.
                            self.PredictedDeviation.append(self.gamma * np.abs(self.series[i] - self.result[i]) 
                                                           + (1-self.gamma)*self.PredictedDeviation[-1])

                        self.UpperBond.append(self.result[-1] + 
                                              self.scaling_factor * 
                                              self.PredictedDeviation[-1])

                        self.LowerBond.append(self.result[-1] - 
                                              self.scaling_factor * 
                                              self.PredictedDeviation[-1])

                        self.Smooth.append(smooth)
                        self.Trend.append(trend)
                        self.Season.append(seasonals[i%self.slen])


            def timeseriesCVscore(params, series, loss_function=mean_squared_error
                                  , slen=season
                                 ):
                """
                    Returns error on CV  

                    params - vector of parameters for optimization
                    series - dataset with timeseries
                    slen - season length for Holt-Winters model
                """
                # errors array
                errors = []

                values = series.values
                alpha, beta, gamma = params

                # set the number of folds for cross-validation
                tscv = TimeSeriesSplit(n_splits=3) 

                # iterating over folds, train model on each, forecast and calculate error
                for train, test in tscv.split(values):

                    model = HoltWinters(series=values[train], slen=slen, 
                                        alpha=alpha, beta=beta, gamma=gamma, n_preds=len(test))
                    model.triple_exponential_smoothing()

                    predictions = model.result[-len(test):]
                    actual = values[test]
                    error = loss_function(predictions, actual)
                    errors.append(error)

                return np.mean(np.array(errors))




            ## Train test and mode
            test_len = round(season*0.8)

            data = df1[pred][:-test_len] # leave some data for testing

            # initializing model parameters alpha, beta and gamma
            x = [0, 0, 0] 

            # Minimizing the loss function 
            opt = minimize(timeseriesCVscore, x0=x, 
                           args=(data, mean_absolute_percentage_error)
                           , method="TNC"
                           , bounds = ((0, 1), (0, 1), (0, 1))
                          )

            # Take optimal values...
            alpha_final, beta_final, gamma_final = opt.x
            print(alpha_final, beta_final, gamma_final)

            # ...and train the model with them, forecasting for the next 50 observations
            model = HoltWinters(data
                                ,slen = season
                                ,alpha = alpha_final, 
                                beta = beta_final, 
                                gamma = gamma_final, 
                                n_preds = 50, scaling_factor = 3)
            model.triple_exponential_smoothing()




            def plotHoltWinters(series, plot_intervals=False, plot_anomalies=False):
                """
                    series - dataset with timeseries
                    plot_intervals - show confidence intervals
                    plot_anomalies - show anomalies 
                """

                plt.figure(figsize=(15, 5))
                plt.plot(model.result, label = "Model")
                plt.plot(series.values, label = "Actual")
                error = mean_absolute_percentage_error(series.values, model.result[:len(series)])
                plt.title("Mean Absolute Percentage Error: {0:.2f}%".format(error))

                if plot_anomalies:
                    anomalies = np.array([np.NaN]*len(series))
                    anomalies[series.values<model.LowerBond[:len(series)]] = \
                        series.values[series.values<model.LowerBond[:len(series)]]
                    anomalies[series.values>model.UpperBond[:len(series)]] = \
                        series.values[series.values>model.UpperBond[:len(series)]]
                    plt.plot(anomalies, "o", markersize=10, label = "Anomalies")

                if plot_intervals:
                    plt.plot(model.UpperBond, "r--", alpha=0.5, label = "Up/Low confidence")
                    plt.plot(model.LowerBond, "r--", alpha=0.5)
                    plt.fill_between(x=range(0,len(model.result)), y1=model.UpperBond, 
                                     y2=model.LowerBond, alpha=0.2, color = "grey")    

                plt.vlines(len(series), ymin=min(model.LowerBond), ymax=max(model.UpperBond), linestyles='dashed')
                plt.axvspan(len(series)-test_len, len(model.result), alpha=0.3, color='lightgrey')
                plt.grid(True)
                plt.axis('tight')
                plt.legend(loc="best", fontsize=13)
                plt.show()
                st.pyplot()

                global mape
                mape = []
                mape.append(['Holt Winter Method', error])

                global mape_df
                mape_df = pd.DataFrame(data=mape, columns=['Model', 'MAPE%'])
                mape_df.drop_duplicates(inplace=True)


            plotHoltWinters(df1[pred])
            st.dataframe(mape_df)
            return mape_df