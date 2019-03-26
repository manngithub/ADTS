import numpy as np
#import scipy as sc
import pandas as pd
#import seaborn as sns
from scipy import signal
from scipy.signal import butter
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import matplotlib
matplotlib.rc('xtick', labelsize=12)
matplotlib.rc('ytick', labelsize=12)

def filter_design(filter_param, data, signal_to_plot, figsize, signals):
    """
    This function design a filter as per user specifications and generate filtered response of data
    
    :type filter_param: dictionary object
    :param filter_param: type: bandpass/lowpass, freq: [low, high] for bandpass filter OR single threshold for lowpass filter, order: filter complexity, fs: sampling rate
    
    :type data: pandas dataframe
    :param data: dataframe that contains sensors measurements
    
    :type signal_to_plot: string
    :param signal_to_plot: names of the signal (column in dataframe: data) to be plotted with its filtered response
    
    :type figsize: tuple
    :param figsize: (width, height) of the figure containing two subplots (filter kernel and original signal with its filtered response)
    
    :type signals: list
    :param signals: list of signals names (column in dataframe data) for which filtered response is required
    
    :return: filtered response of the user specified signals
    """
    # Filter design
    f = filter_param['freq'] # frequency band 
    order = filter_param['order'] # filter order
    
    fs = filter_param['fs'] 
    if filter_param['type'] == 'bandpass':
        f_low = f[0]/(0.5*fs)
        f_high = f[1]/(0.5*fs)
        b, a = butter(order, [f_low,f_high],filter_param['type'])
    elif filter_param['type'] == 'lowpass':
        f = filter_param['freq']
        f_cut = f/(0.5*fs)
        b, a = butter(order, f_cut,filter_param['type'])
    else:
        print ('Attention: Only low or band pass filters are available this time')
        return pd.DataFrame()
    w, h = signal.freqz(b, a, worN = 512*20) # bode plot - frequency response
    
    # plot filter response
    plt.figure(figsize=figsize)
    plt.subplot(2,1,1)
    x_data = w*0.5/(np.pi)
    y_data = abs(h) # 20*np.log10(abs(h)) #
    plt.semilogx(x_data, y_data)
    plt.grid(True,which="both",ls="-",color='y')
    if filter_param['type'] == 'bandpass':
        plt.axvline(f[0], color='green')
        plt.axvline(f[1], color='green')
    elif filter_param['type'] == 'lowpass':
        plt.axvline(f, color='green')
    plt.title('Filter Kernel',fontsize=14)
    plt.xlabel('Frequency',fontsize=12)
    plt.ylabel('Gain',fontsize=12)
    
    # Filter Data
    filter_response = pd.DataFrame(index = data.index)
    for col in signals:
        s = data[col]
        filter_response[col] = signal.filtfilt(b,a,s)
    
    plt.subplot(2,1,2)
    plt.plot(data[signal_to_plot],label=signal_to_plot)
    plt.plot(filter_response[signal_to_plot],label='its filtered response')
    plt.title('Filtered Response',fontsize=14)
    plt.xlabel('Time',fontsize=12)
    plt.legend(loc='best',fontsize=12)
    plt.tight_layout()
    plt.show()
    return filter_response

def df_peak(filter_response, column, threshold):
    """
    This function highlight the time instances where filtered_response of specified signal has amplitude less than the specified threshold
    
    :type filter_response: pandas dataframe
    :param filter_response: dataframe of filtered response of original signals
    
    :type column: string
    :param column: signal name for which plot is generated
    
    :type threshold: float
    :param threshold: threshold to classify the signal as normal (amplitude above threshold) or abnormal (amplitude below threshold)
    
    :return: dataframe that has value 0 when filtered response amplitude is above threshold else 1
    """
    y = filter_response[column]
    col_response = 'Peak_Value_' + column
    y_temp = pd.DataFrame(y.values,index = filter_response.index,columns = [col_response])
    peak_locations = argrelextrema(y_temp.values, np.greater)[0]
    peaks = y_temp.iloc[peak_locations]
    col_peak_loc = 'Peak_Time_' + column
    peaks[col_peak_loc] = peaks.index
    peaks['event'] = 0
    peaks['event'][peaks[col_response] < threshold] = 1
    return peaks

def ground_events_selection(any_event, window_for_ground_event_check):
    """
    This function identifies ground truth events
    
    :type any_event: pandas dataframe
    :param any_event: dataframe that has signal values = 1 for abnormal event else 0
    
    :type window_for_ground_event_check: integer
    :param window_for_ground_event_check: if any signal is abnormal at any time then other signals are checked in this window size before/after that time
    
    :return: ground_events dataframe that has column ground_event_flag = 1 for ground truth (shared by all signals) else 0
    """
    ground_events = any_event.copy()
    signal_columns = ground_events.columns[ground_events.columns.str.contains('signal')]
    events_index = ground_events[ground_events[signal_columns].any(axis = 1)].index # where any_event = 1 for any signal
    ground_events['counter'] = range(ground_events.shape[0])
    ground_events['ground_event_flag'] = 0
    
    for ind in events_index:
        if np.sum(ground_events.loc[ind, signal_columns] == 1) > 1: # already more than 1 signal is having event
            ground_events.loc[ind, 'ground_event_flag'] = 1
        else: # check within a window of 10 (-5 to +5) if another signal has event
            signal_with_1 = np.argmax(ground_events.loc[ind, signal_columns] == 1)
            counter_value = ground_events.loc[ground_events.index == ind,'counter'].values[0]
            # this for loop covers the time before 
            for i in range(window_for_ground_event_check):
                # if any time within window_for_ground_event_check there is another signal has event then mark it as ground event
                if ground_events[(ground_events['counter'] == counter_value-i-1)].loc[:,signal_columns != signal_with_1].values.sum() > 0:
                    ground_events.loc[ind, 'ground_event_flag'] = 1
                    break
            # this for loop covers the time after
            if ground_events.loc[ind, 'ground_event_flag'] != 1: # if we didn't get the ground_event_flag before then check after
                for i in range(window_for_ground_event_check):
                    # if any time within window_for_ground_event_check there is another signal has event then mark it as ground event
                    if ground_events[(ground_events['counter'] == counter_value+i+1)].loc[:,signal_columns != signal_with_1].values.sum() > 0:
                        ground_events.loc[ind, 'ground_event_flag'] = 1
                        break
    return ground_events

def ground_truth_identification(data, filter_response, threshold, window_for_ground_event_check, signal_to_plot, figsize):
    """
    This function identifies the ground truth which is shared by all the signals in the data. It uses the filtered resonse and a user defined threshold to first detect the events that are not normal operations
    
    :type data: pandas dataframe
    :param data: dataframe that contains sensors measurements
    
    :type filter_response: pandas dataframe
    :param filter_response: dataframe for all the signals that has (for any signal) value 0 when filtered response amplitude is above threshold else 1
    
    :type threshold: float
    :param threshold: threshold to classify the signal as normal (amplitude above threshold) or abnormal (amplitude below threshold)
    
    :type window_for_ground_event_check: integer
    :param window_for_ground_event_check: if any signal is abnormal at any time then other signals are checked in this window size before/after that time
    
    :type signal_to_plot: string
    :param signal_to_plot: names of the signal for which different characteristics are plotted for demonstration
    
    :type figsize: tuple
    :param figsize: (width, height) of the figure containing five subplots 
    
    :return: any_event dataframe that has signal values = 1 for abnormal event else 0; ground_events dataframe that has column ground_event_flag = 1 for ground truth (shared by all signals) else 0
    """
    # data: dataframe
    # filter_response: dataframe of filtered data
    # threshold to detect the event
    
    # events: data frame with columns as signals and for each signal at any time the value = 1 when there is an event else 0
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5,1, gridspec_kw = {'height_ratios':[2,2,0.25,1,0.25]}, figsize=figsize,sharex=True)

    ax1.plot(data[signal_to_plot],label=signal_to_plot)
    ax1.plot(filter_response[signal_to_plot],label='its filtered response')
    ax1.set_title('Filtered Response',fontsize=14)
    ax1.legend(loc='best',fontsize=12)
    
    peaks = df_peak(filter_response, signal_to_plot, threshold)
    col_response = 'Peak_Value_' + signal_to_plot
    ax2.plot(peaks[col_response], 'k-', label = 'Peaks profile')
    ax2.plot(filter_response.index,threshold*np.ones(len(filter_response.index)),'r-')
    ax2.set_title("Peak profile of %s and a threshold" %signal_to_plot, fontsize=14)
    
    events = peaks[peaks['event'] == 1]
    ax3.plot(events.index, np.ones(len(events.index)),'b.')
    ax3.set_ylim(0.5,1.5)
    ax3.set_title("Events for %s (flag = 1, else 0)" %signal_to_plot, fontsize=14)
    
    signal_numbers = []
    any_event = pd.DataFrame(index = filter_response.index, columns = filter_response.columns.values).fillna(0)
    for column in filter_response.columns.values:
        peaks = df_peak(filter_response, column, threshold)
        events = peaks[peaks['event'] == 1]
        any_event.loc[events.index, column] = 1
        signal_number = int(column[6:])
        signal_numbers.append(signal_number)
        ax4.plot(events.index, signal_number*np.ones(len(events.index)),'.')
    ax4.set_yticks(signal_numbers)
    ax4.set_title("Locations of events for all signals (Y-axis)", fontsize=14)
    
    # here we select the common location of events from all the signals
    ground_events = ground_events_selection(any_event, window_for_ground_event_check)
    ax5.plot(ground_events.index, ground_events['ground_event_flag'],'.')
    ax5.set_title("Ground Truth (flag = 1, else 0)", fontsize=14)
    ax5.set_ylim(0.5,1.5)
    ax5.set_xlabel('Time',fontsize=12)
    plt.show()
    
    return any_event, ground_events
    
    
    
    