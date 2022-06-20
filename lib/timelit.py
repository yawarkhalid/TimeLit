#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings                                  # `do not disturbe` mode
warnings.filterwarnings('ignore')

import numpy as np                               # vectors and matrices
import pandas as pd                              # tables and data manipulations

import matplotlib.pyplot as plt                  # plots
from matplotlib.pyplot import show, draw, ion
get_ipython().run_line_magic('matplotlib', 'inline')

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
    """
        Calculate average of last n observations
    """
    return np.average(series[-n:])


# In[2]:


def run_all():
    
    import warnings                                  # `do not disturb' mode
    warnings.filterwarnings('ignore')
    
    FILE_NAME = input('input your csv filename: ')
    pred= input('enter name of predictor column: ')
    ds= input ('enter name of datetime column: ')

    dfx = pd.read_csv(FILE_NAME, parse_dates=[ds])
    cols= [ds, pred]
    df = dfx[cols].copy()
    df[ds] = pd.to_datetime(df[ds])
    df.set_index(ds, inplace=True)

    plt.figure(figsize=(15, 7))
    plt.plot(df[pred])
    plt.title('%s'%pred)
    plt.grid(True)
    plt.show()

    print('number of observations in the timeseries', len(df.index))

    dw=input('Do you wish to downsample your series:? [y / n]')

    if dw == 'y' or dw =='Y' or dw =='yes' or dw =='Yes' or dw =='YES':

        rs=input('To which order do you want to downsample your time series?: \n\nH = Hourly\nD = Daily\nW = Weekly\nM = Monthly\nQ = Quarterly\nY = Yearly\n\n')
        agg = (input('how do you want to aggregate your predictor?: \n\nmean = By mean\nsum = By sum\n\n'))

        if agg == 'sum':
            df = df.resample(rs).sum()

        if agg == 'mean':
            df = df.resample(rs).mean()

        plt.figure(figsize=(15, 7))
        plt.plot(df[pred])
        plt.title('%s'%pred)
        plt.grid(True)
        plt.show()

        print('number of observations in the timeseries after downsampling', len(df.index))

    else:
        pass

    df.to_csv('downsampled.csv')

    season = int(input('observe and enter number of observations/season: '))

    #######============########


    #duplicating df for our first model HOLTWINTER


    df1 = df


    def moving_average(series, n):
        return np.average(series[-n:])


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
        plt.show()

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


    def timeseriesCVscore(params, series, loss_function=mean_squared_error, slen=season):
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
    data = df1[pred][:-20] # leave some data for testing

    # initializing model parameters alpha, beta and gamma
    x = [0, 0, 0] 

    # Minimizing the loss function 
    opt = minimize(timeseriesCVscore, x0=x, 
                   args=(data, mean_squared_log_error), 
                   method="TNC", bounds = ((0, 1), (0, 1), (0, 1))
                  )

    # Take optimal values...
    alpha_final, beta_final, gamma_final = opt.x
    print(alpha_final, beta_final, gamma_final)

    # ...and train the model with them, forecasting for the next 50 observations
    model = HoltWinters(data, slen = season, 
                        alpha = alpha_final, 
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

        plt.figure(figsize=(20, 10))
        plt.plot(model.result, label = "Model")
        plt.plot(series.values, label = "Actual")
        error = mean_absolute_percentage_error(series.values, model.result[:len(series)])
        plt.title("Mean Absolute Percentage Error: {0:.2f}%".format(error))

        if plot_anomalies:
            anomalies = np.array([np.NaN]*len(series))
            anomalies[series.values<model.LowerBond[:len(series)]] =                 series.values[series.values<model.LowerBond[:len(series)]]
            anomalies[series.values>model.UpperBond[:len(series)]] =                 series.values[series.values>model.UpperBond[:len(series)]]
            plt.plot(anomalies, "o", markersize=10, label = "Anomalies")

        if plot_intervals:
            plt.plot(model.UpperBond, "r--", alpha=0.5, label = "Up/Low confidence")
            plt.plot(model.LowerBond, "r--", alpha=0.5)
            plt.fill_between(x=range(0,len(model.result)), y1=model.UpperBond, 
                             y2=model.LowerBond, alpha=0.2, color = "grey")    

        plt.vlines(len(series), ymin=min(model.LowerBond), ymax=max(model.UpperBond), linestyles='dashed')
        plt.axvspan(len(series)-20, len(model.result), alpha=0.3, color='lightgrey')
        plt.grid(True)
        plt.axis('tight')
        plt.legend(loc="best", fontsize=13)
        plt.show()

        global mape
        mape = [['Holt Winter Method', error]]

        global mape_df
        mape_df = pd.DataFrame(data=mape, columns=['Model', 'MAPE%'])
        mape_df.drop_duplicates(inplace=True)


    plotHoltWinters(df1[pred])
    display (mape_df)
    
    ###############################
    ##### SARIMA STARTS HERE ######
    ###############################
    
    df2 = df.copy()

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


    sz = input('does seasonality still exist in your timeseries?[y / n] : ')

    if sz == 'y' or sz =='Y' or sz =='yes' or sz =='Yes' or sz =='YES':
        print ('differencing for seasonality.....')
        df_diff = df_diff - df_diff.shift(season)
        count_s += 1
        tsplot(df_diff[season+1:], lags=(season*3))

        more = input('do you want to shift by one more lag to remove seasonality?[y / n]:')

        while more == 'y' or more == 'yes' or more == 'Yes' or more == 'YES' or more == 'Y':
            df_diff = df_diff - df_diff.shift(1)
            count_s += 1
            tsplot(df_diff[season+1:], lags=(season*3))
            more = input('do you want to shift by one more lag to remove seasonality?[y / n]:')


    if sz == 'n' or sz =='no' or sz =='N' or sz =='NO' or sz =='No':
        pass


    #print ('p values after adjusting for seasonality and or stationary: ', adf2)
    print ('Differenced for stationary:', count_d, 'times.')
    print ('Differenced for seasonality:', count_s, 'times.')


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



    print('''

      Ｄｅｔｅｒｍｉｎｉｎｇ　ｐａｒａｍｅｔｅｒ　ｖａｌｕｅｓ:

      𝑝 : To determine the initial 𝑝, you need to look at the PACF plot and find the biggest 
          significant lag after which most other lags become insignificant.

      𝑞 : The initial value of 𝑞 can be found on the ACF plot with the same logic as for 𝑝.

      𝑑 : The order of integration. This is simply the number of nonseasonal differences needed 
          to make the series stationary. In our case, our model has auto differenced based on ADF
          and has automatically taken a value of 𝑑.

      𝑠 : This is responsible for seasonality and equals the season period length of the series.
          We have already input this in the start.

      𝑃 : order of autoregression for the seasonal component of the model, which can be derived from PACF. 
          But you need to look at the number of significant lags, which are the multiples of the season 
          period length. For example, if the period equals 24 and we see the 24-th and 48-th lags are 
          significant in the PACF, that means the initial 𝑃 should be 2.

      𝑄 : Similar logic as 𝑃 but using the ACF plot instead.

      𝐷 : The order of seasonal integration. This can be equal to 1 or 0, depending on whether seasonal 
          differeces were applied or not. Our model has automatically taken this value based on whether
          we differenced or not.

      𝑮𝒊𝒗𝒆𝒏 𝒕𝒉𝒆 𝒂𝒃𝒐𝒗𝒆 𝒎𝒆𝒕𝒉𝒐𝒅, 𝒊𝒏𝒑𝒖𝒕 𝒂𝒏 𝒆𝒔𝒕𝒊𝒎𝒂𝒕𝒆𝒅 𝒓𝒂𝒏𝒈𝒆 𝒇𝒐𝒓 𝒕𝒉𝒆 𝒂𝒃𝒐𝒗𝒆 𝒗𝒂𝒍𝒖𝒆𝒔. 𝑩𝒂𝒔𝒆𝒅 𝒐𝒏 𝒑𝒓𝒐𝒗𝒊𝒅𝒆𝒅 𝒓𝒂𝒏𝒈𝒆𝒔 𝒐𝒖𝒓 𝒎𝒐𝒅𝒆𝒍 𝒘𝒊𝒍𝒍 
      𝒂𝒖𝒕𝒐-𝒅𝒆𝒕𝒆𝒓𝒎𝒊𝒏𝒆 𝒕𝒉𝒆 𝒃𝒆𝒔𝒕 𝒑𝒂𝒓𝒂𝒎𝒆𝒕𝒆𝒓𝒔 𝒇𝒐𝒓 𝒇𝒐𝒓𝒆𝒄𝒂𝒔𝒕𝒊𝒏𝒈. 

      𝑩𝒆𝒘𝒂𝒓𝒆!!: 𝒕𝒉𝒆 𝒍𝒂𝒓𝒈𝒆𝒓 𝒕𝒉𝒆 𝒓𝒂𝒏𝒈𝒆𝒔, 𝒕𝒉𝒆 𝒉𝒊𝒈𝒉𝒆𝒓 𝒕𝒉𝒆 𝒓𝒖𝒏𝒕𝒊𝒎𝒆 𝒇𝒐𝒓 𝒇𝒊𝒏𝒅𝒊𝒏𝒈 𝒕𝒉𝒆 𝒃𝒆𝒔𝒕 𝒑𝒂𝒓𝒂𝒎𝒆𝒕𝒆𝒓𝒔 𝒂𝒔 𝒊𝒕 𝒘𝒊𝒍𝒍 𝒓𝒖𝒏 𝒂𝒏𝒅 𝒕𝒆𝒔𝒕 𝒂𝒍𝒍 
      𝒄𝒐𝒎𝒃𝒊𝒏𝒂𝒕𝒊𝒐𝒏𝒔.


      ''')

    p1 = int(input('starting range for p: '))
    p2 = int(input('ending range for p: '))
    #d1 = int(input('input d = 0 if series has not been differenced, otherwise 1:  '))
    q1 = int(input('starting range for q: '))
    q2 = int(input('ending range for q: '))
    P1 = int(input('starting range for P: '))
    P2 = int(input('ending range for P: '))      
    #D1 = int(input('input D = 0 if series has not been differenced, otherwise 1:  '))
    Q1 = int(input('starting range for Q: '))
    Q2 = int(input('ending range for Q: '))          


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


    print('Optimising SARIMA for best parameters....')
    result_table = optimizeSARIMA(parameters_list, d, D, s, df2[pred])


    # set the parameters that give the lowest AIC
    p, q, P, Q = result_table.parameters[0]

    best_model=sm.tsa.statespace.SARIMAX(df2[pred], order=(p, d, q), 
                                            seasonal_order=(P, D, Q, s)).fit(disp=-1)
    print(best_model.summary())


    ###############################


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

        plt.figure(figsize=(15, 7))
        plt.title("Mean Absolute Percentage Error: {0:.2f}%".format(error))
        plt.plot(forecast, color='r', label="model")
        plt.axvspan(data.index[-1], forecast.index[-1], alpha=0.5, color='lightgrey')
        plt.plot(data.actual, label="actual")
        plt.legend()
        plt.grid(True)
        plt.show()

        global mape
        mape.append(['SARIMA', error])

        global mape_df
        mape_df = pd.DataFrame(data=mape, columns=['Model', 'MAPE%'])
        mape_df.drop_duplicates(inplace=True)

    plotSARIMA(df2, best_model, 50)
    display(mape_df)
    
    
    ###############################################################
    ######## FEATURE ENGINEERING METHODS START HERE ###########
    ###############################################################
    
    # Linear Regression
    # 
    
    
    # Creating a copy of the initial datagrame to make various transformations 
    df3 = df.copy()
    data = pd.DataFrame(df.copy())
    data.columns = ["y"]
    
    # Adding the lag 
    
    print('''
    
    ***** ::: GUIDE FOR SELECTION OF LAGS ::: ******
    
    Increasing the lag, say, up to 6, will allow the model to make predictions 6 steps ahead; however it will 
    use data observed 6 steps back. 
    
    If something fundamentally changes the series during that unobserved period, the model will not catch 
    these changes and will return forecasts with a large error. 
    
    Therefore, during the initial lag selection, one has to find a balance between the optimal prediction 
    quality and the length of the forecasting horizon.
    
    ''')
    
    st_lag=int(input('starting lag from t-x stamps?:'))
    ed_lag=int(input('ending lag from t-x stamps?:'))
    
    # Adding the lag of the target variable from 6 steps back up to 24
    for i in range(6, 25):
        data["lag_{}".format(i)] = data.y.shift(i)
        
    data.head(7)

    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_val_score

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
        
        global mape
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
        
    
    print('\n========== LINEAR REGRESSION FORECASTING ==========\n')
    plotModelResults(lr, plot_intervals=True)
    # plotCoefficients(lr)
    
    ### WITH DATE / HOUR TREND ####
    
    data.index = pd.to_datetime(data.index)
    data["hour"] = data.index.hour
    data["weekday"] = data.index.weekday
    data['is_weekend'] = data.weekday.isin([5,6])*1
    data.tail()
    
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
        
        global mape
        mape.append(['Linear Regression with feature engineering', error])

        global mape_df
        mape_df = pd.DataFrame(data=mape, columns=['Model', 'MAPE%'])
        mape_df.drop_duplicates(inplace=True)


    print('\n========= LINEAR REGRESSION WITH FEATURE ENGINEERING ==========\n')
    plotModelResults_2(lr, X_train=X_train_scaled, X_test=X_test_scaled, plot_intervals=True)
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
    
    X_train, X_test, y_train, y_test =    prepareData(df3[pred], lag_start= st_lag, lag_end=ed_lag, test_size=0.3, target_encoding=False)

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

            global mape
            mape.append(['Ridge regularization', error])

            global mape_df
            mape_df = pd.DataFrame(data=mape, columns=['Model', 'MAPE%'])
            mape_df.drop_duplicates(inplace=True)
    
    
    print ("\n========= FORECASTING WITH RIDGE REGULARIZATION ==========\n ")
    plotModelResults_3(ridge, 
                     X_train=X_train_scaled, 
                     X_test=X_test_scaled, 
                     plot_intervals=True, plot_anomalies=True)
    
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

            global mape
            mape.append(['Lasso regularization', error])

            global mape_df
            mape_df = pd.DataFrame(data=mape, columns=['Model', 'MAPE%'])
            mape_df.drop_duplicates(inplace=True)
    
    
    print ("\n========= FORECASTING WITH LASSO REGULARIZATION ===========\n ")
    plotModelResults_4(lasso, 
                     X_train=X_train_scaled, 
                     X_test=X_test_scaled, 
                     plot_intervals=True, plot_anomalies=True)
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

                global mape
                mape.append(['XG Boost', error])

                global mape_df
                mape_df = pd.DataFrame(data=mape, columns=['Model', 'MAPE%'])
                mape_df.drop_duplicates(inplace=True)
    
    
    print("========== FORECASTING WITH XG BOOST ===========")
    plotModelResults_xgb(xgb, 
                 X_train=X_train_scaled, 
                 X_test=X_test_scaled, 
                 plot_intervals=True, plot_anomalies=True)
    
    display (mape_df)
    
    
    ######################################
    ########## LSTM STARTS HERE ##########
    ######################################
    
    
    test_size = float(input('test size/between 0 and 1: '))
    epochs = int(input('Number of epochs you want the model to run: '))
    
    print('''
        
        !!PLEASE NOTE HIGHER THE NUMBER OF EPOCHS, HIGHER THE TIME AND RESOURCES THE MODELS NEEDS TO TRAIN!
        
        ''')
    
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
    
    
    ## Compiling LSTM
    model = Sequential()

    model.add(LSTM(128,activation = 'LeakyReLU', input_shape= (n_input, n_feature), return_sequences=True))
    model.add(LSTM(128, activation='LeakyReLU', return_sequences=True))
    model.add(LSTM(128, activation='LeakyReLU', return_sequences=False))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    
    display(model.summary())
    
    ## Fitting model
    
    model.fit_generator(train_generator,epochs= epochs)
    
    my_loss= model.history.history['loss']
    plt.plot(range(len(my_loss)),my_loss)
    plt.show()
    
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

        global mape
        mape.append(['LSTM', error])

        global mape_df
        mape_df = pd.DataFrame(data=mape, columns=['Model', 'MAPE%'])
        mape_df.drop_duplicates(inplace=True)

    plot_lstm(error,test)

    ### EVALUATION
    
    def model_eval(error_df):
        df_acc = error_df
        #df_acc = pd.concat(frames)
        df_acc.sort_values('MAPE%', ascending= True, inplace=True)

        plt.figure(figsize=(12,6))
        sns.barplot(x = 'Model' , y= 'MAPE%' , data = df_acc)
        plt.show()
        
        return df_acc

    model_acc = model_eval(mape_df)
    
    display(model_acc)

    return model_acc


# # Individual Functions

# In[3]:


mape = []


# In[4]:


def model_eval(error_df):
    df_acc = error_df
    #df_acc = pd.concat(frames)
    df_acc.sort_values('MAPE%', ascending= True, inplace=True)

    plt.figure(figsize=(12,6))
    sns.barplot(x = 'Model' , y= 'MAPE%' , data = df_acc)
    plt.show()

    return df_acc


# In[5]:


def holt_winters():
    
    import warnings                                  # `do not disturb' mode
    warnings.filterwarnings('ignore')
    
    FILE_NAME = input('input your csv filename: ')
    pred= input('enter name of predictor column: ')
    ds= input ('enter name of datetime column: ')

    dfx = pd.read_csv(FILE_NAME, parse_dates=[ds])
    cols= [ds, pred]
    df = dfx[cols].copy()
    df[ds] = pd.to_datetime(df[ds])
    df.set_index(ds, inplace=True)

    plt.figure(figsize=(15, 7))
    plt.plot(df[pred])
    plt.title('%s'%pred)
    plt.grid(True)
    plt.show()

    print('number of observations in the timeseries', len(df.index))

    dw=input('Do you wish to downsample your series:? [y / n]')

    if dw == 'y' or dw =='Y' or dw =='yes' or dw =='Yes' or dw =='YES':

        rs=input('To which order do you want to downsample your time series?: \n\nH = Hourly\nD = Daily\nW = Weekly\nM = Monthly\nQ = Quarterly\nY = Yearly\n\n')
        agg = (input('how do you want to aggregate your predictor?: \n\nmean = By mean\nsum = By sum\n\n'))

        if agg == 'sum':
            df = df.resample(rs).sum()

        if agg == 'mean':
            df = df.resample(rs).mean()

        plt.figure(figsize=(15, 7))
        plt.plot(df[pred])
        plt.title('%s'%pred)
        plt.grid(True)
        plt.show()

        print('number of observations in the timeseries after downsampling', len(df.index))

    else:
        pass

    df.to_csv('downsampled.csv')

    season = int(input('observe and enter number of observations/season: '))
    
    #######============########


    #duplicating df for our first model HOLTWINTER


    df1 = df


    def moving_average(series, n):
        return np.average(series[-n:])


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
        plt.show()

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


    def timeseriesCVscore(params, series, loss_function=mean_squared_error, slen=season):
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
    data = df1[pred][:-20] # leave some data for testing

    # initializing model parameters alpha, beta and gamma
    x = [0, 0, 0] 

    # Minimizing the loss function 
    opt = minimize(timeseriesCVscore, x0=x, 
                   args=(data, mean_squared_log_error), 
                   method="TNC", bounds = ((0, 1), (0, 1), (0, 1))
                  )

    # Take optimal values...
    alpha_final, beta_final, gamma_final = opt.x
    print(alpha_final, beta_final, gamma_final)

    # ...and train the model with them, forecasting for the next 50 observations
    model = HoltWinters(data, slen = season, 
                        alpha = alpha_final, 
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

        plt.figure(figsize=(20, 10))
        plt.plot(model.result, label = "Model")
        plt.plot(series.values, label = "Actual")
        error = mean_absolute_percentage_error(series.values, model.result[:len(series)])
        plt.title("Mean Absolute Percentage Error: {0:.2f}%".format(error))

        if plot_anomalies:
            anomalies = np.array([np.NaN]*len(series))
            anomalies[series.values<model.LowerBond[:len(series)]] =                 series.values[series.values<model.LowerBond[:len(series)]]
            anomalies[series.values>model.UpperBond[:len(series)]] =                 series.values[series.values>model.UpperBond[:len(series)]]
            plt.plot(anomalies, "o", markersize=10, label = "Anomalies")

        if plot_intervals:
            plt.plot(model.UpperBond, "r--", alpha=0.5, label = "Up/Low confidence")
            plt.plot(model.LowerBond, "r--", alpha=0.5)
            plt.fill_between(x=range(0,len(model.result)), y1=model.UpperBond, 
                             y2=model.LowerBond, alpha=0.2, color = "grey")    

        plt.vlines(len(series), ymin=min(model.LowerBond), ymax=max(model.UpperBond), linestyles='dashed')
        plt.axvspan(len(series)-20, len(model.result), alpha=0.3, color='lightgrey')
        plt.grid(True)
        plt.axis('tight')
        plt.legend(loc="best", fontsize=13)
        plt.show()

        global mape
        mape.append(['Holt Winter Method', error])

        global mape_df
        mape_df = pd.DataFrame(data=mape, columns=['Model', 'MAPE%'])
        mape_df.drop_duplicates(inplace=True)


    plotHoltWinters(df1[pred])
    display (mape_df)


# In[6]:


def SARIMA():
    
    import warnings                                  # `do not disturb' mode
    warnings.filterwarnings('ignore')
    
    FILE_NAME = input('input your csv filename: ')
    pred= input('enter name of predictor column: ')
    ds= input ('enter name of datetime column: ')

    dfx = pd.read_csv(FILE_NAME, parse_dates=[ds])
    cols= [ds, pred]
    df = dfx[cols].copy()
    df[ds] = pd.to_datetime(df[ds])
    df.set_index(ds, inplace=True)

    plt.figure(figsize=(15, 7))
    plt.plot(df[pred])
    plt.title('%s'%pred)
    plt.grid(True)
    plt.show()

    print('number of observations in the timeseries', len(df.index))

    dw=input('Do you wish to downsample your series:? [y / n]')

    if dw == 'y' or dw =='Y' or dw =='yes' or dw =='Yes' or dw =='YES':

        rs=input('To which order do you want to downsample your time series?: \n\nH = Hourly\nD = Daily\nW = Weekly\nM = Monthly\nQ = Quarterly\nY = Yearly\n\n')
        agg = (input('how do you want to aggregate your predictor?: \n\nmean = By mean\nsum = By sum\n\n'))

        if agg == 'sum':
            df = df.resample(rs).sum()

        if agg == 'mean':
            df = df.resample(rs).mean()

        plt.figure(figsize=(15, 7))
        plt.plot(df[pred])
        plt.title('%s'%pred)
        plt.grid(True)
        plt.show()

        print('number of observations in the timeseries after downsampling', len(df.index))

    else:
        pass

    df.to_csv('downsampled.csv')

    season = int(input('observe and enter number of observations/season: '))
    
    
    ###############################
    ##### SARIMA STARTS HERE ######
    ###############################
    
    df2 = df.copy()

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


    sz = input('does seasonality still exist in your timeseries?[y / n] : ')

    if sz == 'y' or sz =='Y' or sz =='yes' or sz =='Yes' or sz =='YES':
        print ('differencing for seasonality.....')
        df_diff = df_diff - df_diff.shift(season)
        count_s += 1
        tsplot(df_diff[season+1:], lags=(season*3))

        more = input('do you want to shift by one more lag to remove seasonality?[y / n]:')

        while more == 'y' or more == 'yes' or more == 'Yes' or more == 'YES' or more == 'Y':
            df_diff = df_diff - df_diff.shift(1)
            count_s += 1
            tsplot(df_diff[season+1:], lags=(season*3))
            more = input('do you want to shift by one more lag to remove seasonality?[y / n]:')


    if sz == 'n' or sz =='no' or sz =='N' or sz =='NO' or sz =='No':
        pass


    #print ('p values after adjusting for seasonality and or stationary: ', adf2)
    print ('Differenced for stationary:', count_d, 'times.')
    print ('Differenced for seasonality:', count_s, 'times.')


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



    print('''

      Ｄｅｔｅｒｍｉｎｉｎｇ　ｐａｒａｍｅｔｅｒ　ｖａｌｕｅｓ:

      𝑝 : To determine the initial 𝑝, you need to look at the PACF plot and find the biggest 
          significant lag after which most other lags become insignificant.

      𝑞 : The initial value of 𝑞 can be found on the ACF plot with the same logic as for 𝑝.

      𝑑 : The order of integration. This is simply the number of nonseasonal differences needed 
          to make the series stationary. In our case, our model has auto differenced based on ADF
          and has automatically taken a value of 𝑑.

      𝑠 : This is responsible for seasonality and equals the season period length of the series.
          We have already input this in the start.

      𝑃 : order of autoregression for the seasonal component of the model, which can be derived from PACF. 
          But you need to look at the number of significant lags, which are the multiples of the season 
          period length. For example, if the period equals 24 and we see the 24-th and 48-th lags are 
          significant in the PACF, that means the initial 𝑃 should be 2.

      𝑄 : Similar logic as 𝑃 but using the ACF plot instead.

      𝐷 : The order of seasonal integration. This can be equal to 1 or 0, depending on whether seasonal 
          differeces were applied or not. Our model has automatically taken this value based on whether
          we differenced or not.

      𝑮𝒊𝒗𝒆𝒏 𝒕𝒉𝒆 𝒂𝒃𝒐𝒗𝒆 𝒎𝒆𝒕𝒉𝒐𝒅, 𝒊𝒏𝒑𝒖𝒕 𝒂𝒏 𝒆𝒔𝒕𝒊𝒎𝒂𝒕𝒆𝒅 𝒓𝒂𝒏𝒈𝒆 𝒇𝒐𝒓 𝒕𝒉𝒆 𝒂𝒃𝒐𝒗𝒆 𝒗𝒂𝒍𝒖𝒆𝒔. 𝑩𝒂𝒔𝒆𝒅 𝒐𝒏 𝒑𝒓𝒐𝒗𝒊𝒅𝒆𝒅 𝒓𝒂𝒏𝒈𝒆𝒔 𝒐𝒖𝒓 𝒎𝒐𝒅𝒆𝒍 𝒘𝒊𝒍𝒍 
      𝒂𝒖𝒕𝒐-𝒅𝒆𝒕𝒆𝒓𝒎𝒊𝒏𝒆 𝒕𝒉𝒆 𝒃𝒆𝒔𝒕 𝒑𝒂𝒓𝒂𝒎𝒆𝒕𝒆𝒓𝒔 𝒇𝒐𝒓 𝒇𝒐𝒓𝒆𝒄𝒂𝒔𝒕𝒊𝒏𝒈. 

      𝑩𝒆𝒘𝒂𝒓𝒆!!: 𝒕𝒉𝒆 𝒍𝒂𝒓𝒈𝒆𝒓 𝒕𝒉𝒆 𝒓𝒂𝒏𝒈𝒆𝒔, 𝒕𝒉𝒆 𝒉𝒊𝒈𝒉𝒆𝒓 𝒕𝒉𝒆 𝒓𝒖𝒏𝒕𝒊𝒎𝒆 𝒇𝒐𝒓 𝒇𝒊𝒏𝒅𝒊𝒏𝒈 𝒕𝒉𝒆 𝒃𝒆𝒔𝒕 𝒑𝒂𝒓𝒂𝒎𝒆𝒕𝒆𝒓𝒔 𝒂𝒔 𝒊𝒕 𝒘𝒊𝒍𝒍 𝒓𝒖𝒏 𝒂𝒏𝒅 𝒕𝒆𝒔𝒕 𝒂𝒍𝒍 
      𝒄𝒐𝒎𝒃𝒊𝒏𝒂𝒕𝒊𝒐𝒏𝒔.


      ''')

    p1 = int(input('starting range for p: '))
    p2 = int(input('ending range for p: '))
    #d1 = int(input('input d = 0 if series has not been differenced, otherwise 1:  '))
    q1 = int(input('starting range for q: '))
    q2 = int(input('ending range for q: '))
    P1 = int(input('starting range for P: '))
    P2 = int(input('ending range for P: '))      
    #D1 = int(input('input D = 0 if series has not been differenced, otherwise 1:  '))
    Q1 = int(input('starting range for Q: '))
    Q2 = int(input('ending range for Q: '))          


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


    print('Optimising SARIMA for best parameters....')
    result_table = optimizeSARIMA(parameters_list, d, D, s, df2[pred])


    # set the parameters that give the lowest AIC
    p, q, P, Q = result_table.parameters[0]

    best_model=sm.tsa.statespace.SARIMAX(df2[pred], order=(p, d, q), 
                                            seasonal_order=(P, D, Q, s)).fit(disp=-1)
    print(best_model.summary())


    ###############################


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

        plt.figure(figsize=(15, 7))
        plt.title("Mean Absolute Percentage Error: {0:.2f}%".format(error))
        plt.plot(forecast, color='r', label="model")
        plt.axvspan(data.index[-1], forecast.index[-1], alpha=0.5, color='lightgrey')
        plt.plot(data.actual, label="actual")
        plt.legend()
        plt.grid(True)
        plt.show()

        global mape
        mape.append(['SARIMA', error])

        global mape_df
        mape_df = pd.DataFrame(data=mape, columns=['Model', 'MAPE%'])
        mape_df.drop_duplicates(inplace=True)

    plotSARIMA(df2, best_model, 50)
    display(mape_df)
    


# In[7]:


def FE_methods():
    
    import warnings                                  # `do not disturb' mode
    warnings.filterwarnings('ignore')
    
    FILE_NAME = input('input your csv filename: ')
    pred= input('enter name of predictor column: ')
    ds= input ('enter name of datetime column: ')

    dfx = pd.read_csv(FILE_NAME, parse_dates=[ds])
    cols= [ds, pred]
    df = dfx[cols].copy()
    df[ds] = pd.to_datetime(df[ds])
    df.set_index(ds, inplace=True)

    plt.figure(figsize=(15, 7))
    plt.plot(df[pred])
    plt.title('%s'%pred)
    plt.grid(True)
    plt.show()

    print('number of observations in the timeseries', len(df.index))

    dw=input('Do you wish to downsample your series:? [y / n]')

    if dw == 'y' or dw =='Y' or dw =='yes' or dw =='Yes' or dw =='YES':

        rs=input('To which order do you want to downsample your time series?: \n\nH = Hourly\nD = Daily\nW = Weekly\nM = Monthly\nQ = Quarterly\nY = Yearly\n\n')
        agg = (input('how do you want to aggregate your predictor?: \n\nmean = By mean\nsum = By sum\n\n'))

        if agg == 'sum':
            df = df.resample(rs).sum()

        if agg == 'mean':
            df = df.resample(rs).mean()

        plt.figure(figsize=(15, 7))
        plt.plot(df[pred])
        plt.title('%s'%pred)
        plt.grid(True)
        plt.show()

        print('number of observations in the timeseries after downsampling', len(df.index))

    else:
        pass

    df.to_csv('downsampled.csv')

    #season = int(input('observe and enter number of observations/season: '))
    
    ###############################################################
    ######## FEATURE ENGINEERING METHODS START HERE ###########
    ###############################################################
    
    # Linear Regression
    # 
    
    
    # Creating a copy of the initial datagrame to make various transformations 
    df3 = df.copy()
    data = pd.DataFrame(df.copy())
    data.columns = ["y"]
    
    # Adding the lag 
    
    print('''
    
    ***** ::: GUIDE FOR SELECTION OF LAGS ::: ******
    
    Increasing the lag, say, up to 6, will allow the model to make predictions 6 steps ahead; however it will 
    use data observed 6 steps back. 
    
    If something fundamentally changes the series during that unobserved period, the model will not catch 
    these changes and will return forecasts with a large error. 
    
    Therefore, during the initial lag selection, one has to find a balance between the optimal prediction 
    quality and the length of the forecasting horizon.
    
    ''')
    
    st_lag=int(input('starting lag from t-x stamps?:'))
    ed_lag=int(input('ending lag from t-x stamps?:'))
    
    # Adding the lag of the target variable from 6 steps back up to 24
    for i in range(6, 25):
        data["lag_{}".format(i)] = data.y.shift(i)
        
    data.head(7)

    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_val_score

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
        
        global mape
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
        
    
    print('\n========== LINEAR REGRESSION FORECASTING ==========\n')
    plotModelResults(lr, plot_intervals=True)
    # plotCoefficients(lr)
    
    ### WITH DATE / HOUR TREND ####
    
    data.index = pd.to_datetime(data.index)
    data["hour"] = data.index.hour
    data["weekday"] = data.index.weekday
    data['is_weekend'] = data.weekday.isin([5,6])*1
    data.tail()
    
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
        
        global mape
        mape.append(['Linear Regression with feature engineering', error])

        global mape_df
        mape_df = pd.DataFrame(data=mape, columns=['Model', 'MAPE%'])
        mape_df.drop_duplicates(inplace=True)


    print('\n========= LINEAR REGRESSION WITH FEATURE ENGINEERING ==========\n')
    plotModelResults_2(lr, X_train=X_train_scaled, X_test=X_test_scaled, plot_intervals=True)
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
    
    X_train, X_test, y_train, y_test =    prepareData(df3[pred], lag_start= st_lag, lag_end=ed_lag, test_size=0.3, target_encoding=False)

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

            global mape
            mape.append(['Ridge regularization', error])

            global mape_df
            mape_df = pd.DataFrame(data=mape, columns=['Model', 'MAPE%'])
            mape_df.drop_duplicates(inplace=True)
    
    
    print ("\n========= FORECASTING WITH RIDGE REGULARIZATION ==========\n ")
    plotModelResults_3(ridge, 
                     X_train=X_train_scaled, 
                     X_test=X_test_scaled, 
                     plot_intervals=True, plot_anomalies=True)
    
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

            global mape
            mape.append(['Lasso regularization', error])

            global mape_df
            mape_df = pd.DataFrame(data=mape, columns=['Model', 'MAPE%'])
            mape_df.drop_duplicates(inplace=True)
    
    
    print ("\n========= FORECASTING WITH LASSO REGULARIZATION ===========\n ")
    plotModelResults_4(lasso, 
                     X_train=X_train_scaled, 
                     X_test=X_test_scaled, 
                     plot_intervals=True, plot_anomalies=True)
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

                global mape
                mape.append(['XG Boost', error])

                global mape_df
                mape_df = pd.DataFrame(data=mape, columns=['Model', 'MAPE%'])
                mape_df.drop_duplicates(inplace=True)
    
    
    print("========== FORECASTING WITH XG BOOST ===========")
    plotModelResults_xgb(xgb, 
                 X_train=X_train_scaled, 
                 X_test=X_test_scaled, 
                 plot_intervals=True, plot_anomalies=True)
    
    display (mape_df)


# In[8]:


def LSTM():
    
    import warnings                                  # `do not disturb' mode
    warnings.filterwarnings('ignore')
    
    FILE_NAME = input('input your csv filename: ')
    pred= input('enter name of predictor column: ')
    ds= input ('enter name of datetime column: ')

    dfx = pd.read_csv(FILE_NAME, parse_dates=[ds])
    cols= [ds, pred]
    df = dfx[cols].copy()
    df[ds] = pd.to_datetime(df[ds])
    df.set_index(ds, inplace=True)

    plt.figure(figsize=(15, 7))
    plt.plot(df[pred])
    plt.title('%s'%pred)
    plt.grid(True)
    plt.show()

    print('number of observations in the timeseries', len(df.index))

    dw=input('Do you wish to downsample your series:? [y / n]')

    if dw == 'y' or dw =='Y' or dw =='yes' or dw =='Yes' or dw =='YES':

        rs=input('To which order do you want to downsample your time series?: \n\nH = Hourly\nD = Daily\nW = Weekly\nM = Monthly\nQ = Quarterly\nY = Yearly\n\n')
        agg = (input('how do you want to aggregate your predictor?: \n\nmean = By mean\nsum = By sum\n\n'))

        if agg == 'sum':
            df = df.resample(rs).sum()

        if agg == 'mean':
            df = df.resample(rs).mean()

        plt.figure(figsize=(15, 7))
        plt.plot(df[pred])
        plt.title('%s'%pred)
        plt.grid(True)
        plt.show()

        print('number of observations in the timeseries after downsampling', len(df.index))

    else:
        pass

    df.to_csv('downsampled.csv')

    #season = int(input('observe and enter number of observations/season: '))
    
    
    ######################################
    ########## LSTM STARTS HERE ##########
    ######################################
    
    
    test_size = float(input('test size/between 0 and 1: '))
    epochs = int(input('Number of epochs you want the model to run: '))
    
    print('''
        
        !!PLEASE NOTE HIGHER THE NUMBER OF EPOCHS, HIGHER THE TIME AND RESOURCES THE MODELS NEEDS TO TRAIN!
        
        ''')
    
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
    
    
    ## Compiling LSTM
    model = Sequential()

    model.add(LSTM(128,activation = 'LeakyReLU', input_shape= (n_input, n_feature), return_sequences=True))
    model.add(LSTM(128, activation='LeakyReLU', return_sequences=True))
    model.add(LSTM(128, activation='LeakyReLU', return_sequences=False))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    
    display(model.summary())
    
    ## Fitting model
    
    model.fit_generator(train_generator,epochs= epochs)
    
    my_loss= model.history.history['loss']
    plt.plot(range(len(my_loss)),my_loss)
    plt.show()
    
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

        global mape
        mape.append(['LSTM', error])

        global mape_df
        mape_df = pd.DataFrame(data=mape, columns=['Model', 'MAPE%'])
        mape_df.drop_duplicates(inplace=True)

    plot_lstm(error,test)
    display (mape_df)

