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


def ML_models_st(df, season, pred, ds):
    
    df3 = df
    data = pd.DataFrame(df.copy())
    data.columns = ["y"]
    
    pred = pred
    ds = ds
    season = season
    
    st.markdown('### **Machine Learning Models**')
    
    # Adding the lag 
    
    st.markdown('''
    
    ##### **GUIDE FOR SELECTION OF TIMESERIES LAGS**
    
    Shifting the series  n  steps back, we get a feature column where the current value of time series is aligned with its value at time  t−n . If we make a 1 lag shift and train a model on that feature, the model will be able to forecast 1 step ahead from having observed the current state of the series. Increasing the lag, say, up to 6, will allow the model to make predictions 6 steps ahead; however it will use data observed 6 steps back. If something fundamentally changes the series during that unobserved period, the model will not catch these changes and will return forecasts with a large error. 
    
    **_Therefore, during the initial lag selection, one has to find a balance between the optimal prediction quality and the length of the forecasting horizon._**
    
    ''')
    
    st_lag= int(st.number_input('Start lagging from x timestamp?'
                                ,min_value = 1.0, step=1.0))
    ed_lag= int(st.number_input('End lagging on x timestamp?'
                                ,min_value = 1.0, step=1.0))
    
    
    if st.button('Forecast with ML Models'):
        # Adding the lag of the target variable from 6 steps back up to 24
        for i in range(st_lag, ed_lag):
            data["lag_{}".format(i)] = data.y.shift(i)

        st.markdown('##### **Data with Lags**')
        st.dataframe(data.head(10))

        # for time-series cross-validation set 5 folds 
        tscv = TimeSeriesSplit(n_splits=5)

        def timeseries_train_test_split(X, y, test_size):
            """
                Perform train-test split with respect to time series structure
            """

            # get the index after which test set starts
            test_index = int(len(X)*(1-test_size))

            X_train = X.iloc[:test_index]
            y_train = y.iloc[:test_index]
            X_test = X.iloc[test_index:]
            y_test = y.iloc[test_index:]

            return X_train, X_test, y_train, y_test

        y = data.dropna().y
        X = data.dropna().drop(['y'], axis=1)

        # reserve 30% of data for testing
        X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size=0.3)


        # machine learning in two lines
        lr = LinearRegression()
        lr.fit(X_train, y_train)


        def plotModelResults(model, X_train=X_train, X_test=X_test, plot_intervals=False, plot_anomalies=False):
            """
                Plots modelled vs fact values, prediction intervals and anomalies

            """

            prediction = model.predict(X_test)

            plt.figure(figsize=(15, 7))
            plt.plot(prediction, "g", label="prediction", linewidth=2.0)
            plt.plot(y_test.values, label="actual", linewidth=2.0)

            if plot_intervals:
                cv = cross_val_score(model, X_train, y_train, 
                                            cv=tscv, 
                                            scoring="neg_mean_absolute_error")
                mae = cv.mean() * (-1)
                deviation = cv.std()

                scale = 1.96
                lower = prediction - (mae + scale * deviation)
                upper = prediction + (mae + scale * deviation)

                plt.plot(lower, "r--", label="upper bond / lower bond", alpha=0.5)
                plt.plot(upper, "r--", alpha=0.5)

                if plot_anomalies:
                    anomalies = np.array([np.NaN]*len(y_test))
                    anomalies[y_test<lower] = y_test[y_test<lower]
                    anomalies[y_test>upper] = y_test[y_test>upper]
                    plt.plot(anomalies, "o", markersize=10, label = "Anomalies")

            error = mean_absolute_percentage_error(prediction, y_test)
            plt.title("Mean absolute percentage error {0:.2f}%".format(error))
            plt.legend(loc="best")
            plt.tight_layout()
            plt.grid(True)
            plt.show()
            st.pyplot()

            global mape
            mape = []
            mape.append(['Linear Regression', error])

            global mape_df
            mape_df = pd.DataFrame(data=mape, columns=['Model', 'MAPE%'])
            mape_df.drop_duplicates(inplace=True)

        def plotCoefficients(model):

                #Plots sorted coefficient values of the model

            coefs = pd.DataFrame(model.coef_, X_train.columns)
            coefs.columns = ["coef"]
            coefs["abs"] = coefs.coef.apply(np.abs)
            coefs = coefs.sort_values(by="abs", ascending=False).drop(["abs"], axis=1)

            plt.figure(figsize=(15, 7))
            coefs.coef.plot(kind='bar')
            plt.grid(True, axis='y')
            plt.hlines(y=0, xmin=0, xmax=len(coefs), linestyles='dashed')
            plt.show
            st.pyplot()


        st.markdown('##### **LINEAR REGRESSION FORECASTING**')
        plotModelResults(lr, plot_intervals=False)
        # plotCoefficients(lr)

        ### WITH DATE / HOUR TREND ####

        data.index = pd.to_datetime(data.index)
        data["hour"] = data.index.hour
        data["weekday"] = data.index.weekday
        data['is_weekend'] = data.weekday.isin([5,6])*1

        st.markdown('##### **Dataframe with feature engineering (Weekday, Hour, Is_Weekend)**')
        st.dataframe(data.tail())

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()

        y = data.dropna().y
        X = data.dropna().drop(['y'], axis=1)

        X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size=0.3)

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        lr = LinearRegression()
        lr.fit(X_train_scaled, y_train)


        def plotModelResults_2(model, X_train=X_train, X_test=X_test, plot_intervals=False, plot_anomalies=False):
            """
                Plots modelled vs fact values, prediction intervals and anomalies

            """

            prediction = model.predict(X_test)

            plt.figure(figsize=(15, 7))
            plt.plot(prediction, "g", label="prediction", linewidth=2.0)
            plt.plot(y_test.values, label="actual", linewidth=2.0)

            if plot_intervals:
                cv = cross_val_score(model, X_train, y_train, 
                                            cv=tscv, 
                                            scoring="neg_mean_absolute_error")
                mae = cv.mean() * (-1)
                deviation = cv.std()

                scale = 1.96
                lower = prediction - (mae + scale * deviation)
                upper = prediction + (mae + scale * deviation)

                plt.plot(lower, "r--", label="upper bond / lower bond", alpha=0.5)
                plt.plot(upper, "r--", alpha=0.5)

                if plot_anomalies:
                    anomalies = np.array([np.NaN]*len(y_test))
                    anomalies[y_test<lower] = y_test[y_test<lower]
                    anomalies[y_test>upper] = y_test[y_test>upper]
                    plt.plot(anomalies, "o", markersize=10, label = "Anomalies")

            error = mean_absolute_percentage_error(prediction, y_test)
            plt.title("Mean absolute percentage error {0:.2f}%".format(error))
            plt.legend(loc="best")
            plt.tight_layout()
            plt.grid(True)
            plt.show()
            st.pyplot()

            global mape
            #mape = []
            mape.append(['Linear Regression with feature engineering', error])

            global mape_df
            mape_df = pd.DataFrame(data=mape, columns=['Model', 'MAPE%'])
            mape_df.drop_duplicates(inplace=True)


        st.markdown('##### **LINEAR REGRESSION WITH FEATURE ENGINEERING**')
        plotModelResults_2(lr, X_train=X_train_scaled, X_test=X_test_scaled, plot_intervals=False)
        #plotCoefficients(lr)


        ### WITH REGULARIZATION ###

        def prepareData(series, lag_start, lag_end, test_size, target_encoding=False):
            """
                series: pd.DataFrame
                    dataframe with timeseries

                lag_start: int
                    initial step back in time to slice target variable 
                    example - lag_start = 1 means that the model 
                              will see yesterday's values to predict today

                lag_end: int
                    final step back in time to slice target variable
                    example - lag_end = 4 means that the model 
                              will see up to 4 days back in time to predict today

                test_size: float
                    size of the test dataset after train/test split as percentage of dataset

                target_encoding: boolean
                    if True - add target averages to the dataset

            """

            # copy of the initial dataset
            data = pd.DataFrame(series.copy())
            data.columns = ["y"]

            # lags of series
            for i in range(lag_start, lag_end):
                data["lag_{}".format(i)] = data.y.shift(i)

            # datetime features
            data.index = pd.to_datetime(data.index)
            data["hour"] = data.index.hour
            data["weekday"] = data.index.weekday
            data['is_weekend'] = data.weekday.isin([5,6])*1

            if target_encoding:
                # calculate averages on train set only
                test_index = int(len(data.dropna())*(1-test_size))
                data['weekday_average'] = list(map(code_mean(data[:test_index], 'weekday', "y").get, data.weekday))
                data["hour_average"] = list(map(code_mean(data[:test_index], 'hour', "y").get, data.hour))

                # frop encoded variables 
                data.drop(["hour", "weekday"], axis=1, inplace=True)

            # train-test split
            y = data.dropna().y
            X = data.dropna().drop(['y'], axis=1)
            X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size=test_size)

            return X_train, X_test, y_train, y_test

        #st.write(df3)
        X_train, X_test, y_train, y_test =\
        prepareData(df3[pred], lag_start= st_lag, lag_end=ed_lag, test_size=0.3, target_encoding=False)

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        from sklearn.linear_model import LassoCV, RidgeCV

        ridge = RidgeCV(cv=tscv)
        ridge.fit(X_train_scaled, y_train)

        def plotModelResults_3(model, X_train=X_train, X_test=X_test, plot_intervals=False, plot_anomalies=False):
                """
                    Plots modelled vs fact values, prediction intervals and anomalies

                """

                prediction = model.predict(X_test)

                plt.figure(figsize=(15, 7))
                plt.plot(prediction, "g", label="prediction", linewidth=2.0)
                plt.plot(y_test.values, label="actual", linewidth=2.0)

                if plot_intervals:
                    cv = cross_val_score(model, X_train, y_train, 
                                                cv=tscv, 
                                                scoring="neg_mean_absolute_error")
                    mae = cv.mean() * (-1)
                    deviation = cv.std()

                    scale = 1.96
                    lower = prediction - (mae + scale * deviation)
                    upper = prediction + (mae + scale * deviation)

                    plt.plot(lower, "r--", label="upper bond / lower bond", alpha=0.5)
                    plt.plot(upper, "r--", alpha=0.5)

                    if plot_anomalies:
                        anomalies = np.array([np.NaN]*len(y_test))
                        anomalies[y_test<lower] = y_test[y_test<lower]
                        anomalies[y_test>upper] = y_test[y_test>upper]
                        plt.plot(anomalies, "o", markersize=10, label = "Anomalies")

                error = mean_absolute_percentage_error(prediction, y_test)
                plt.title("Mean absolute percentage error {0:.2f}%".format(error))
                plt.legend(loc="best")
                plt.tight_layout()
                plt.grid(True)
                plt.show()
                st.pyplot()

                global mape
                #mape = []
                mape.append(['Ridge regularization', error])

                global mape_df
                mape_df = pd.DataFrame(data=mape, columns=['Model', 'MAPE%'])
                mape_df.drop_duplicates(inplace=True)

        st.markdown ("##### **FORECASTING WITH RIDGE REGRESSION**")

        plotModelResults_3(ridge, 
                         X_train=X_train_scaled, 
                         X_test=X_test_scaled, 
                         plot_intervals=False, plot_anomalies=False)


        ##LASSO

        lasso = LassoCV(cv=tscv)
        lasso.fit(X_train_scaled, y_train)

        def plotModelResults_4(model, X_train=X_train, X_test=X_test, plot_intervals=False, plot_anomalies=False):
                """
                    Plots modelled vs fact values, prediction intervals and anomalies

                """

                prediction = model.predict(X_test)

                plt.figure(figsize=(15, 7))
                plt.plot(prediction, "g", label="prediction", linewidth=2.0)
                plt.plot(y_test.values, label="actual", linewidth=2.0)

                if plot_intervals:
                    cv = cross_val_score(model, X_train, y_train, 
                                                cv=tscv, 
                                                scoring="neg_mean_absolute_error")
                    mae = cv.mean() * (-1)
                    deviation = cv.std()

                    scale = 1.96
                    lower = prediction - (mae + scale * deviation)
                    upper = prediction + (mae + scale * deviation)

                    plt.plot(lower, "r--", label="upper bond / lower bond", alpha=0.5)
                    plt.plot(upper, "r--", alpha=0.5)

                    if plot_anomalies:
                        anomalies = np.array([np.NaN]*len(y_test))
                        anomalies[y_test<lower] = y_test[y_test<lower]
                        anomalies[y_test>upper] = y_test[y_test>upper]
                        plt.plot(anomalies, "o", markersize=10, label = "Anomalies")

                error = mean_absolute_percentage_error(prediction, y_test)
                plt.title("Mean absolute percentage error {0:.2f}%".format(error))
                plt.legend(loc="best")
                plt.tight_layout()
                plt.grid(True)
                plt.show()
                st.pyplot()

                global mape
                #mape=[]
                mape.append(['Lasso regularization', error])

                global mape_df
                mape_df = pd.DataFrame(data=mape, columns=['Model', 'MAPE%'])
                mape_df.drop_duplicates(inplace=True)


        st.markdown("##### **FORECASTING WITH LASSO REGRESSION**")
        plotModelResults_4(lasso, 
                         X_train=X_train_scaled, 
                         X_test=X_test_scaled, 
                         plot_intervals=False, plot_anomalies=False)
        #plotCoefficients(lasso)


        from xgboost import XGBRegressor 

        xgb = XGBRegressor()
        xgb.fit(X_train_scaled, y_train)

        def plotModelResults_xgb(model, X_train=X_train, X_test=X_test, plot_intervals=False, plot_anomalies=False):
                    """
                        Plots modelled vs fact values, prediction intervals and anomalies

                    """

                    prediction = model.predict(X_test)

                    plt.figure(figsize=(15, 7))
                    plt.plot(prediction, "g", label="prediction", linewidth=2.0)
                    plt.plot(y_test.values, label="actual", linewidth=2.0)

                    if plot_intervals:
                        cv = cross_val_score(model, X_train, y_train, 
                                                    cv=tscv, 
                                                    scoring="neg_mean_absolute_error")
                        mae = cv.mean() * (-1)
                        deviation = cv.std()

                        scale = 1.96
                        lower = prediction - (mae + scale * deviation)
                        upper = prediction + (mae + scale * deviation)

                        plt.plot(lower, "r--", label="upper bond / lower bond", alpha=0.5)
                        plt.plot(upper, "r--", alpha=0.5)

                        if plot_anomalies:
                            anomalies = np.array([np.NaN]*len(y_test))
                            anomalies[y_test<lower] = y_test[y_test<lower]
                            anomalies[y_test>upper] = y_test[y_test>upper]
                            plt.plot(anomalies, "o", markersize=10, label = "Anomalies")

                    error = mean_absolute_percentage_error(prediction, y_test)
                    plt.title("Mean absolute percentage error {0:.2f}%".format(error))
                    plt.legend(loc="best")
                    plt.tight_layout()
                    plt.grid(True)
                    plt.show()
                    st.pyplot()

                    global mape
                    #mape = []
                    mape.append(['XG Boost', error])

                    global mape_df
                    mape_df = pd.DataFrame(data=mape, columns=['Model', 'MAPE%'])
                    mape_df.drop_duplicates(inplace=True)


        st.markdown("##### **FORECASTING WITH XG BOOST**")
        plotModelResults_xgb(xgb, 
                     X_train=X_train_scaled, 
                     X_test=X_test_scaled, 
                     plot_intervals=False, plot_anomalies=False)

        st.markdown ('##### **Mean Absoulte Percentage Errors %**')
        st.dataframe (mape_df)
        return mape_df