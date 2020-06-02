# Spectrum analysis.
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.signal as sig
import commpy.filters as com

# Display method
def display():
    plt.show()

# EVM Meter.
def evmMeter(signalIn, symbolsIn, fs, freqMix, debugMode = False):
    # Generating reference signal
    ts = 1e-6
    osr = int(fs * ts)
    numberSamples = int(8 * osr)
    time, pulseShape = com.rrcosfilter(numberSamples, 0.5, ts, fs)
    symbolsUp = np.zeros(symbolsIn.size * osr) + 1j * np.zeros(symbolsIn.size * osr)
    symbolsUp[0::osr] = symbolsIn
    phaseInc = 2 * np.pi * freqMix / fs * np.ones(symbolsUp.size + pulseShape.size - 1)
    signalRef = np.convolve(symbolsUp, pulseShape) * np.exp(1j * phaseInc.cumsum())

    #timeMix = np.arange(0, symbolsUp.size + pulseShape.size - 1) * (1 / fs)
    #signalRef = np.convolve(symbolsUp, pulseShape) * np.exp(2j * np.pi * freqMix * timeMix)

    signalRefLevel = np.sqrt(np.abs(np.mean(signalRef * signalRef.conj())))
    if (debugMode):
        signalPlotting(signalRef, fs, 'Signal Ref')
        signalPlotting(signalIn, fs, 'Signal Ref')

    if (debugMode):
        fig1, (ax1, ax12) = plt.subplots(figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k', ncols=1, nrows=2,
                                         sharex=False)
        ax1.plot(signalRef.real, 'b', label='real')
        ax1.plot(signalRef.imag, 'r', label='imag')
        ax1.set_title(f'Signal Ref Level = {signalRefLevel} with length {signalRef.size}')
        ax1.legend()
        ax12.plot(signalIn.real, 'b', label='real')
        ax12.plot(signalIn.imag, 'r', label='imag')
        ax12.set_title(f'Signal In with length {signalIn.size}')
        ax12.legend()

    # Calculate EVM

    # Check lenghts are ok
    if signalIn.size < signalRef.size:
        raise SyntaxError('ERROR: DSPFunctions::evmMeter - Signal is shorter than reference.')

    crossCorrelSignal = np.correlate(signalIn, signalRef, 'full')
    index = np.arange(-np.max([signalIn.size, signalRef.size]) + 1, np.max([signalIn.size, signalRef.size]), 1)
    indexAligned = index[-crossCorrelSignal.size:]
    indexAligned = indexAligned[int(indexAligned.size/2):]
    crossCorrelSignal = crossCorrelSignal[int(crossCorrelSignal.size/2):]
    lag = indexAligned[np.abs(crossCorrelSignal).argmax()]
    if (lag < 0) :
        raise SyntaxError('ERROR: DSPFunction::evmMeter - index of correlation is negative.')

    if (debugMode):
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        ax2.plot(signalIn.real, 'b', label='Signal In Real')
        ax2.plot(signalRef.real, 'g', label='Signal Ref Real')
        ax2.plot(indexAligned, np.abs(crossCorrelSignal) / np.abs(crossCorrelSignal).max(), 'r', label='CrossCorrelation')
        ax2.set_title(f'Cross-correlation lag = {lag}')
        ax2.legend()

    # chopping signals
    signalInChop = signalIn[lag:]
    if (debugMode):
        print('Lag = ', lag, 'signalInChop = ', signalInChop.size, 'signalIn = ', signalIn.size)
        print(signalIn[:10].real*1e3)
        print(signalInChop[:10].real*1e3)
    signalLen = np.min([signalInChop.size, signalRef.size])
    if (debugMode):
        print('signalInChop = ', signalInChop.size)
        print('signalLen = ', signalLen)
    signalInChop = signalInChop[:signalLen]
    signalRefChop = signalRef[:signalLen]
    if (debugMode):
        print('signalLen = ', signalLen)
        print(signalIn[:10].real * 1e3)
        print(signalInChop[:10].real * 1e3)

    if (debugMode):
        print('signalInChop = ', signalInChop.size, ' signalRefChop = ', signalRefChop.size)

    if (debugMode):
        fig3, (ax3, ax4) = plt.subplots(figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k', ncols=1, nrows=2, sharex=False)
        signalInChopLevel = np.sqrt(np.abs(np.mean(signalInChop * signalInChop.conj())))
        gain = signalRefLevel / signalInChopLevel
        ax3.plot(signalRefChop.real, 'b', label='Ref real')
        ax3.plot(gain * signalInChop.real, 'r--', label='Signal In real')
        ax3.legend()
        ax3.set_title(f'Signal ref Chopped - length = {signalRefChop.size}')
        ax4.plot(signalRefChop.imag, 'b', label='Ref imag')
        ax4.plot(gain * signalInChop.imag, 'r--', label='Signal In imag')
        ax4.legend()
        ax4.set_title(f'Signal In Chopped - length = {signalInChop.size}')

    # Input Signal level and rotation
    signalInLevel = np.sqrt(np.abs(np.mean(signalInChop * signalInChop.conj())))
    signalInRot = (signalRefLevel / signalInLevel) * np.exp(-1j * np.angle(crossCorrelSignal.max())) * signalInChop
    if (debugMode):
        fig5, ax5 = plt.subplots(figsize=(10, 8))
        ax5.plot(signalInRot.real, 'b', label='Rot Real')
        ax5.plot(signalInRot.imag, 'r', label='Rot Imag')
        ax5.plot(gain*signalInChop.real, 'b--', label='Org Real')
        ax5.plot(gain*signalInChop.imag, 'r--', label='Org Imag')
        ax5.set_title(f'Rot Signal angle {np.angle(crossCorrelSignal.max())}, signalRefLevel = {signalRefLevel}, signalInLevel = {signalInLevel}')
        ax5.legend()

    # Error Vector
    errorVector = signalInRot - signalRefChop
    rmsRef = np.sqrt(np.mean(np.abs(signalRefChop) ** 2))
    evmValue = 20 * np.log10(np.sqrt(np.mean(np.abs(np.abs(errorVector) ** 2))) / rmsRef)
    if (debugMode):
        print ('rmsRef = ', rmsRef)
        fig6, (ax6a, ax6b) = plt.subplots(figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k', ncols=1, nrows=2, sharex=False)
        ax6aa = ax6a.twinx()
        ax6a.plot(signalInRot.real, 'r', label='Rot Real')
        ax6a.plot(signalRef.real, 'b--', label='Ref Real')
        ax6aa.plot(20 * np.log10(errorVector.real / rmsRef), 'k', label='Error')
        ax6aa.legend()
        ax6bb = ax6b.twinx()
        ax6b.plot(signalInRot.imag, 'r', label='Rot Imag')
        ax6b.plot(signalRef.imag, 'b--', label='Ref Imag')
        ax6bb.plot(20 * np.log10(errorVector.imag / rmsRef), 'k', label='Error')
        ax6bb.legend()
        ax6a.set_title(f'EVM = {evmValue}')

    plt.show()

    return (evmValue)

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

def freqResponse(filterCoeff, samplingFreq, label, debugMode = False):
    """
    :type filterCoeff: Coefficients of the Filter
    """
    (wb, Hb) = sig.freqz(filterCoeff)
    if (debugMode):
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
        plt.show()

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


def ConstellationPlot(signal, fs, OSR, label):
    fig, ax = plt.subplots(figsize=(10, 10), dpi=80, facecolor='w')
    for phaseCount in range (0, OSR):
        signalDec = signal[phaseCount::OSR]
        legendLabel = f'Decimation Phase = {phaseCount}'
        ax.scatter(signalDec.real, signalDec.imag, label=legendLabel)

    ax.set_title(f'Scatter Plot - {label}')
    ax.legend()
    ax.grid()
    ax.set_xlabel('Real')
    ax.set_ylabel('Imag')

