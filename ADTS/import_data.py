"""
This module contains functions to deliver the data as requested.
"""
import numpy as np
import pandas as pd
#import random
import warnings 
warnings.filterwarnings('ignore') # ignore warnings
#import matplotlib.pyplot as plt
import os

def import_data(path, datafile, generate_data):
    """
    Generate/Import simulated time series data of sensors' measurements
    
    :type path: string
    :param path: data directory path relative to working file
    
    :type datafile: string
    :param datafile: data file name
    
    :type generate_data: boolean
    :param generate_data: True to generate simulated data, False to import existing data
    
    :return: data in a dataframe
    """
    if generate_data:
        print('[Info] Generating Simulation Data...')
        duration = 30*24*60 # minutes
        time_step = 1 # minute
        t = np.array(range(0,duration,time_step))
        signal_amplitude = 20.0
        noise_amplitude = 1
        time_period = 25
        start_down = 20*60
        end_down = 60*60
        base_value = 500
        trigger_events = [[4080 , 4260], [8150 , 8330], [12000 , 12180], [16100, 16280], [24000, 24180], 
                          [30000,30180], [35050,35230], [40080,40260]]
        #[time_period*240 , time_period*240 + 180], [10080 , 10080 + 180]
        
        # first 5 signals
        
        f1 = signal_amplitude*np.sin(2*np.pi*(1.0/time_period)*t) 
        external_factor = (signal_amplitude/5)*np.sin(2*np.pi*(1.0/(time_period*57.6))*t)
        f1 = f1 + external_factor
        f1[start_down-2:end_down+2] = 0
        # adding trigger_events
        for event_range in trigger_events:
            f1[event_range[0]:event_range[1]] = -signal_amplitude*2 + signal_amplitude*2*np.cos(2*np.pi*(1.0/240)*t[event_range[0]:event_range[1]])
        noise = noise_amplitude*np.random.randn(duration)
        operation_down = 0.45*base_value*(- np.tanh(t-start_down) + np.tanh(t - end_down))
        
        signal = f1 + base_value + operation_down
        signal1 = signal + noise
        signal2 = signal - 10 + noise
        signal3 = np.roll(signal,4) + noise
        signal3[0] = signal3[1]
        signal3 = signal3 + 10
        signal4 = np.roll(signal,3) + noise
        signal4[0] = signal4[1]
        signal5 = np.roll(signal,2) + noise
        signal5[0] = signal5[1]
        
        #plt.figure(1)
        #plt.plot(t,signal1,t,signal2,t,signal3,t,signal4,t,signal5)
        #plt.show()
        
        # next 5 signals
        f1 = signal_amplitude*np.sin(2*np.pi*(1.0/time_period)*t) 
        external_factor = (signal_amplitude/5)*np.sin(2*np.pi*(1.0/(time_period*57.6))*t + 2*np.pi*((time_period*28.8)/(time_period*57.6)))
        f1 = f1 + external_factor
        f1[start_down-2:end_down+2] = 0
        # adding trigger_events
        for event_range in trigger_events:
            f1[event_range[0]:event_range[1]] = -signal_amplitude*2 + signal_amplitude*2*np.cos(2*np.pi*(1.0/240)*t[event_range[0]:event_range[1]])
        noise = noise_amplitude*np.random.randn(duration)
        operation_down = 0.45*base_value*(- np.tanh(t-start_down) + np.tanh(t - end_down))
        
        signal = f1 + base_value + operation_down
        signal6 = signal + noise
        signal7 = signal - 10 + noise
        signal8 = np.roll(signal,4) + noise
        signal8[0] = signal8[1]
        signal8 = signal8 + 10
        signal9 = np.roll(signal,3) + noise
        signal9[0] = signal9[1]
        signal10 = np.roll(signal,2) + noise
        signal10[0] = signal10[1]
        
        #plt.figure(2)
        #plt.plot(t,signal1,t,signal2,t,signal3,t,signal4,t,signal5,t,signal6,t,signal7,t,signal8,t,signal9,t,signal10)
        #plt.show()
        
        clog_events = [[4400, 4600], [8600, 8800],[18000,18200],[21000,21200],[32000,32200],[39000,39200]]
        lag = 5
        # create anomaly events in one or two signals using tanh function...
        
        for duration in clog_events:
            clog_duration = len(range(duration[1] - duration[0]))
            noise_clog = noise_amplitude*np.random.randn(clog_duration)
            signal6 = signal6 + 30.0*(- np.tanh(t-duration[0]) + np.tanh(t - duration[1]))
            signal6[duration[0]+lag:duration[1]-lag] = np.mean(signal6[duration[0]+lag:duration[1]]-lag) + noise_clog[lag:-lag]
        #plt.figure(3)
        #plt.plot(t,signal6)
        #plt.show()
        
        T = pd.to_datetime(t,unit="m") + pd.Timedelta(days=40*365 + 10) 
        data = pd.DataFrame({'signal1':signal1, 'signal2':signal2, 'signal3':signal3, 'signal4':signal4, 'signal5':signal5, 'signal6':signal6, 'signal7':signal7, 'signal8':signal8, 'signal9':signal9, 'signal10':signal10}, index = T)
        #data.to_csv('simulated_data.csv')
    else: # import existing data
        # find the path of current working directory
        select, ignore = os.path.split(os.path.abspath(__file__))
        
        # training data
        print('[Info] Data Loading...')
        data = pd.read_csv(select+'/'+path+'/'+datafile, index_col = 0)
        data.index = pd.to_datetime(data.index)
    return data