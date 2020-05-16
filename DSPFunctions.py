# Spectrum analysis.
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.signal as sig

def display():
    plt.show()

def signalPlotting(signal, samplingFreq, label):
    time = ((np.arange (0, signal.size)) / samplingFreq) * 1e6
    fftSize = np.fmin(np.power(2, 12), signal.size)
    fftSignal = 20 * np.log10(1 / fftSize * np.abs(np.fft.fft(signal, fftSize)))
    fftFreq = np.fft.fftfreq(fftSize) * samplingFreq / 1e6

    fig, (ax1, ax2) = plt.subplots(figsize=(18, 10), dpi=80, facecolor='w', edgecolor='k', ncols=2, nrows=1, sharex=False)
    ax1.plot(time, signal.real, color='b', label='Real')
    ax1.plot(time, signal.imag, color='r', label='Imag')
    ax1.set_ylabel('I/Q Amplitude')
    ax1.set_xlabel('Time [us]')
    ax1.margins(x=0, y=0.1)
    ax1.grid()
    ax1.set_title(f'Time Domain - {label}')
    ax1.legend()

    ax2.plot(np.fft.fftshift(fftFreq), np.fft.fftshift(fftSignal), color='g')
    ax2.set_ylabel('Signal level [dBFS]')
    ax2.set_xlabel('Frequency [MHz]')
    ax1.margins(x=0, y=0.1)
    ax2.grid()
    ax2.set_title(f'Frequency Domain - {label}')

def fftPlotting(signal, samplingFreq, label):
    fftSize=np.power(2,10)
    fftSignal = np.fft.fft(signal,fftSize)
    fftFreq = np.fft.fftfreq(fftSize)
    plt.figure(3)
    plt.plot(fftFreq * samplingFreq/1e6, 20 * np.log10(1 / fftSize * np.abs(fftSignal)))
    plt.title(label)
    plt.xlabel('Frequency [Mhz]')
    plt.ylabel('Frequency Response [dBFS]')
    plt.grid(True)
    plt.show()

def freqResponse(filterCoeff, samplingFreq, label):
    """
    :type filterCoeff: Coefficients of the Filter
    """
    (wb, Hb) = sig.freqz(filterCoeff)
    print('Half Band Filter order N = ', filterCoeff.size-1)
    for count1 in range(filterCoeff.size):
        print(' tap %2d   %3.6f' % (count1, filterCoeff[count1]))
    fig = plt.figure(figsize=(9, 5), dpi=80, facecolor='w', edgecolor='k')
    ax1 = fig.add_subplot(111)
    ax1.plot(wb*samplingFreq/(2*np.pi*1e6), 20*np.log10(np.abs(Hb)))
    ax1.set_ylabel('Magnitude (dB)')
    ax1.set_xlabel('Frequency (MHz)')
    ax1.set_title(label)
    ax1.grid(True)
    #plt.show()

def timePlot(signal, samplingFreq, label):
    """
    ;type timePlot: plot IQ signal.
    :param signal: IQ signal to plot.
    :param samplingFre: Sampling Frequency.
    :param label: label.
    :return:
    """
    fig = plt.figure()
    time = ((np.arange (0, signal.size))/samplingFreq)*1e6
    ax1 = fig.add_subplot(111)
    ax1.plot(time, np.real(signal), 'b')
    ax1.plot(time, np.imag(signal), 'r')
    ax1.set_ylabel('Signal Amplitude')
    ax1.set_xlabel('time (us)')
    ax1.set_title(label)
    ax1.grid(True)
    red_patch = mpatches.Patch(color='red', label='In-Phase, Quadrature')
    ax1.legend(handles=[red_patch])
    plt.show()
