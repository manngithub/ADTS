import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy as sc
from scipy import signal
import pandas as pd
import matplotlib
matplotlib.rc('xtick', labelsize=12)
matplotlib.rc('ytick', labelsize=12)

def plot_data(data, column, figsize, title, kind):
    """
    This function is used to plot the data history 
    
    :type data: pandas dataframe
    :param data: dataframe that contains the signals' measurements in time
    
    :type column: string
    :param column: names of the signals (column in dataframe: data) to be plotted
    
    :type figsize: tuple
    :param figsize: (width, height) of the plot
    
    :type title: string
    :param title: title of the plot
    
    :type kind: string
    :param kind: type of plot (box, line)
    """
    if kind == 'line':
        plt.figure(figsize = figsize)
        plt.plot(data[column], label = column)
        plt.legend(loc='best', fontsize = 12)
        plt.title(title, fontsize = 14)
        plt.xlabel('Time',fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.show()
    elif kind == 'box':
        data[column].plot(kind = kind, figsize = figsize, title = title, fontsize = 12)

def correlation_analysis(data, figsize):
    """
    This function generates the correlation heat map of all measurements in the dataframe data
    
    :type data: pandas dataframe
    :param data: dataframe with measurements for which correlations are calculated
    
    :type figsize: tuple
    :param figsize: (width, height) of the correlation heatmap
    """
    fig, ax = plt.subplots(figsize=figsize) 
    corr = data.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    sns.set(font_scale=1.4)
    sns.heatmap(corr,mask = mask,linecolor='white',linewidth=.2, ax = ax)
    
def fourier_analysis(data, column, T, zoom_in, figsize):
    """
    This function calculates the fast fourier transform (FFT) of specified signal and provides the frequency plot
    
    :type data: pandas dataframe
    :param data: dataframe that contains sensors measurements
    
    :type column: string
    :param column: name of the signal for which FFT is performed
    
    :type T: float
    :param T: sampling time period
    
    :type zoom_in: list
    :param zoom_in: zoom x-axis of the plot for frequency associated with [lowest value, highest value] of time period of signal 
    
    :type figsize: tuple
    :param figsize: (width, height) of the frequency plot
    """
    T = float(T)
    y = data[column]
    N = y.shape[0]
    yf = sc.fft(y) # fourier transform using Tukey algorithm
    xf = np.linspace(0.0,1.0/(2.0*T),N//2) # frequency domain
    yf = 2.0/N*np.abs(yf[0:N//2]) # magnitude of frequency component
    df = pd.DataFrame({'xf':xf,'yf':yf})
    
    zoom_in = map(float,zoom_in) # minutes
    df_selected = df[(df['xf'] > 1.0/zoom_in[1]) & (df['xf'] < 1.0/zoom_in[0])] # zoom in
    
    plt.figure(figsize = figsize)
    plt.plot(df_selected['xf'],df_selected['yf'])
    plt.title('Frequency plot of %s'%(column), fontsize = 14)
    plt.ylabel('Magnitude', fontsize = 12)
    plt.xlabel('Frequency', fontsize = 12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()
    max_value_at = df_selected['xf'][df_selected['yf'].argmax()]
    print 'Peak value in this zoom plot appears at time period : %.2f' %(1.0/max_value_at)
    print 'and Frequency is %.5f' %(max_value_at)
    
def spectrogram_analysis(data, column, figsize, zoom_in, title):
    """
    This function performs the spectrogram analysis for specified signal
    
    :type data: pandas dataframe
    :param data: dataframe that contains sensors measurements
    
    :type column: string
    :param column: name of the signal for which spectrogram is generated
    
    :type figsize: tuple
    :param figsize: (width, height) of the figure containing two subplots (original signal and its spectrogram)
    
    :type zoom_in: list
    :param zoom_in: zoom y-axis of the plot for frequency associated with [lowest value, highest value]  
    
    :type title: list
    :param title: list of two strings that are titles of two subplots
    """
    y = data[column] # signal
    f, t, Sxx = signal.spectrogram(y)
    fmin = zoom_in[0]
    fmax = zoom_in[1]
    fig, ax = plt.subplots(2,1,figsize = figsize)
    ax[0].plot(data[column], label = column)
    ax[0].legend(loc='best', fontsize = 12)
    ax[0].autoscale(enable=True, axis='x', tight=True)
    ax[0].set_title(title[0], fontsize = 14)
    
    ax[1].pcolormesh(t, f[(f>fmin) & (f<fmax)], Sxx[(f>fmin) & (f<fmax),:], cmap='RdBu_r')
    ax[1].set_title(title[1], fontsize = 14)
    ax[1].set_xticks([])
    fig.tight_layout()
    plt.show()