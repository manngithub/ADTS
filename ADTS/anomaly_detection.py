
import numpy as np
#import scipy as sc
import pandas as pd
#import seaborn as sns
#from scipy import signal
#from scipy.signal import butter
import matplotlib.pyplot as plt
#from scipy.signal import argrelextrema
#from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.rc('xtick', labelsize=12)
matplotlib.rc('ytick', labelsize=12)

def train_test_data(data, any_event, sampling, train_test_split_time, cover):
    """
    This function divides the data in train-test data sets and then collect the normal training data by filtering out the data near events in train data set
    
    :type data: pandas dataframe
    :param data: dataframe that contains sensors measurements
    
    :type any_event: pandas dataframe
    :param any_event: dataframe that has signal values = 1 for abnormal event else 0
    
    :type sampling: string
    :param sampling: sampling time increment in measurements - minutes/seconds/milliseconds
    
    :type train_test_split_time: string
    :param train_test_split_time: time in format YYYY-MM-DD HH:MM:SS to split the data in train and test data sets
    
    :type cover: integer
    :param cover: any data point around event that is covered in cover window is filtered from normal training data
    
    :return: three dataframes train_data, test_data, normal_train_data
    """      
    any_event['indicator'] = 0
    # indicator = 1 for the cover window around the detected event, else = 0
    for sig in any_event.columns[any_event.columns.str.contains('signal')]:
        locations = any_event[any_event[sig] == 1]
        for i in range(locations.shape[0]):
            any_event.loc[locations.index[i] - pd.Timedelta(minutes=cover):locations.index[i] + \
                              pd.Timedelta(minutes=cover),'indicator'] = 1
    # train-test split
    train_test_split_time = pd.to_datetime(train_test_split_time)
    train_data_indicator = any_event.loc[any_event.index[0]:train_test_split_time, ['indicator']]
    if sampling == 'minutes':
        test_data_indicator = any_event.loc[train_test_split_time+pd.Timedelta(minutes=1):any_event.index[-1], ['indicator']]
    elif sampling == 'seconds':
        test_data_indicator = any_event.loc[train_test_split_time+pd.Timedelta(seconds=1):any_event.index[-1], ['indicator']]
    elif sampling == 'milliseconds':
        test_data_indicator = any_event.loc[train_test_split_time+pd.Timedelta(milliseconds=1):any_event.index[-1], ['indicator']]
    else:
        print 'Sampling in minutes/ seconds/ milliseconds are considered for this version'
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), []
    
    # train-test data split
    data['indicator'] = any_event['indicator']
    train_data = data.loc[train_data_indicator.index, data.columns]
    test_data = data.loc[test_data_indicator.index, data.columns]
    # collect the normal training data
    normal_train_data = train_data.loc[train_data['indicator'] != 1, train_data.columns[train_data.columns.str.contains('signal')]]
    
    return train_data, test_data, normal_train_data

def hotelling_T2_train(normal_train_data, ifPlot, figsize):
    """
    This function uses the normal training data and train model based on Hotelling T-squared statistics
    
    :type normal_train_data: pandas dataframe
    :param normal_train_data: dataframe of normal training data
    
    :type ifPlot: boolean
    :param ifPlot: True/False to plot the T-squared distribution from normal training data
    
    :type figsize: tuple
    :param figsize: (width, height) of the plot
    
    :return: normal_train_data_mean, normal_train_data_covariance, normal_train_data_T2
    """
    normal_train_data = normal_train_data.iloc[1:]
    normal_train_data_mean = normal_train_data.mean(axis=0)
    normalized_data = normal_train_data - normal_train_data_mean
    normal_train_data_covariance = normalized_data.values.T.dot(normalized_data.values)/(normalized_data.shape[0]-1) 
 
    # Calculate T2 statistics
    T2_statistics = np.zeros(normal_train_data.shape[0],dtype=float)
    
    for i in range(T2_statistics.shape[0]):
        sample = normalized_data.iloc[i].values.reshape(normalized_data.shape[1],1)
        T2_statistics[i] = sample.T.dot(np.linalg.inv(normal_train_data_covariance).dot(sample))
    
    normal_train_data_T2 = normal_train_data.copy()
    normal_train_data_T2['T2_statistics'] = T2_statistics
    
    if ifPlot:
        plt.figure(figsize=figsize)
        plt.hist(T2_statistics,bins=100)
        plt.xlabel('T sqaured distance',fontsize=12)
        plt.ylabel('frequency',fontsize=12)
        plt.show()
    
    return normal_train_data_mean, normal_train_data_covariance, normal_train_data_T2

def anomaly_and_ground_events(test_data_T2, ground_events, window_check, thresholds):
    """
    This function identify the anomaly and filter the ground events
    
    :type test_data_T2: pandas dataframe
    :param test_data_T2: dataframe of T-squared statistics of each signal
    
    :type ground_events: pandas dataframe
    :param ground_events: dataframe where each signal has value = 1 for ground truth event else 0
    
    :type window_check: integer
    :param window_check: if any event detected by T-squared statistics has ground_event within window_check before or after the event, then it is regarded as ground_event
    
    :type thresholds: numpy list
    :param thresholds: [low, high] threshold. If T-squared statistics < low or > high then it is highlighted as event
    
    :return: test_data_T2 dataframe of T-squared statistics on test data. It has column anomaly_and_ground_events = 1 for ground truth event, = 2 for anomaly, = 0 for normal operation
    """
    test_data_T2['anomaly_and_ground_events'] = 0
    events_locations = (test_data_T2['T2_statistics'] < thresholds[0]) | (test_data_T2['T2_statistics'] > thresholds[1])
    test_data_T2.loc[events_locations==True,'anomaly_and_ground_events'] = 2
    test_data_T2['counter'] = range(test_data_T2.shape[0])
    test_data_T2 = pd.concat([test_data_T2, ground_events.loc[test_data_T2.index][['ground_event_flag']]], axis=1)
    
    events_index = test_data_T2[test_data_T2['anomaly_and_ground_events']==2].index
    for ind in events_index:
        counter_value = test_data_T2.loc[test_data_T2.index == ind,'counter'].values[0]
        # check ground_event_flag in window_check time before, if flag = 1, then event is actually a ground truth event
        if test_data_T2.loc[test_data_T2['counter'].isin(range(counter_value,counter_value-window_check,-1)), 'ground_event_flag'].values.sum() > 0:
            test_data_T2.loc[ind,'anomaly_and_ground_events'] = 1
        # check ground_event_flag in next window_check time, if flag = 1, then event is actually a ground truth event
        if test_data_T2.loc[test_data_T2['counter'].isin(range(counter_value,counter_value+window_check)), 'ground_event_flag'].values.sum() > 0:
            test_data_T2.loc[ind,'anomaly_and_ground_events'] = 1
    
    return test_data_T2

def hotelling_T2_test(test_data, ground_events, window_check, normal_train_data_mean, normal_train_data_covariance, ifPlot, figsize, signal_to_plot, thresholds):
    """
    This function calculates the T-squared statistics on test data and flag anomalous events
    :type test_data: pandas dataframe
    :param test_data: dataframe of test dataset
    
    :type ground_events: pandas dataframe
    :param ground_events: dataframe where each signal has value = 1 for ground truth event else 0
    
    :type window_check: integer
    :param window_check: if any event detected by T-squared statistics has ground_event within window_check before or after the event, then it is regarded as ground_event
    
    :type normal_train_data_mean: pandas series
    :param normal_train_data_mean: mean value of signals from normal training data
    
    :type normal_train_data_covariance: numpy array
    :param normal_train_data_covariance: covariance matrix from normal training data
    
    :type ifPlot: boolean
    :param ifPlot: True/False to plot the T-squared distribution from normal training data
    
    :type figsize: tuple
    :param figsize: (width, height) of the figure
    
    :type signal_to_plot: string
    :param signal_to_plot: names of the signals to be plotted with events locations
    
    :type thresholds: numpy list
    :param thresholds: [low, high] threshold. If T-squared statistics < low or > high then it is highlighted as event
    
    :return: test_data_T2 dataframe of T-squared statistics on test data. It has column anomaly_and_ground_events = 1 for ground truth event, = 2 for anomaly, = 0 for normal operation
    """
    # Calculate T2 statistics
    signal_cols = test_data.columns[test_data.columns.str.contains('signal')]
    normalized_test_data = test_data[signal_cols] - normal_train_data_mean
    T2_statistics_test = np.zeros(test_data.shape[0],dtype=float)
    
    for i in range(T2_statistics_test.shape[0]):
        sample = normalized_test_data.iloc[i].values.reshape(normalized_test_data.shape[1],1)
        T2_statistics_test[i] = sample.T.dot(np.linalg.inv(normal_train_data_covariance).dot(sample))
    
    test_data_T2 = test_data.copy()
    test_data_T2['T2_statistics'] = T2_statistics_test
    
    if ifPlot:
        fig, (ax1, ax2) = plt.subplots(2,1, gridspec_kw = {'height_ratios':[1,1]}, figsize=figsize,sharex=True)
        ax1.plot(test_data_T2[signal_to_plot], label = signal_to_plot)
        ax1.set_title("Test Data of %s" %signal_to_plot, fontsize=14)

    test_data_T2 = anomaly_and_ground_events(test_data_T2, ground_events, window_check, thresholds)
    if ifPlot:
        ax2.plot(test_data_T2['anomaly_and_ground_events'],'.')
        ax2.set_title('Normal Operation (flag = 0), Ground Truth (flag = 1) and Anomaly Event (flag = 2)', fontsize=14)
        ax2.set_ylim(- 0.5,2.5)
        plt.tight_layout()
        plt.show()
    
    return test_data_T2