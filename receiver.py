import numpy as np
import DSPFunctions
from scipy import signal as sgn

class Rx:
    # Constructor
    def __init__(self, modulationScheme, IF):
        self.myFs = 32e6
        self.myIF = IF
        self.myModulationSchem = modulationScheme

    def FsOver4(self, signal):
        switcher = {
            0: np.exp(-1j * np.pi * 0 / 2),
            1: np.exp(-1j * np.pi * 1 / 2),
            2: np.exp(-1j * np.pi * 2 / 2),
            3: np.exp(-1j * np.pi * 3 / 2),
        }
        fsOver4Signal = np.zeros(signal.size) + 1j * np.zeros(signal.size)
        for counter in range(0, signal.size - 1):
            fsOver4Signal[counter] = signal[counter] * switcher.get(counter % 4)
        return fsOver4Signal

    def Downsampler(self, signal, downsamplinPhase, order, flatBand, fsDown, label, debugMode=False):
        downSamplingFactor = 2
        downCoeff = sgn.firwin(order + 1, flatBand, fs=fsDown)
        if (debugMode):
            DSPFunctions.freqResponse(downCoeff, fsDown, label)
        filteredSignal = np.convolve(signal, downCoeff)
        outSignal = filteredSignal[downsamplinPhase::downSamplingFactor]
        return outSignal

    def Mixing(self, signal, freqMix, fsMix):
        phase = 2 * np.pi * np.ones(signal.size) * freqMix / fsMix
        mixedSignal = signal * np.exp(-1j * phase.cumsum())
        return mixedSignal

    def Run(self, signal, symbols):

        # Received signal
        evmRx = DSPFunctions.evmMeter(signal, symbols, self.myFs, self.myIF, False)
        DSPFunctions.signalPlotting(signal, self.myFs, f'Receiver. EVM = {evmRx} dB')

        # Mixing down
        mixDownSignal = self.Mixing(signal, self.myIF, self.myFs)
        evmMixDown = DSPFunctions.evmMeter(mixDownSignal, symbols, self.myFs, 0, False)
        DSPFunctions.signalPlotting(mixDownSignal, self.myFs, f'Down IF Mixer. EVM = {evmMixDown} dB')

        # Zeroth downsampler
        down0Signal = self.Downsampler(mixDownSignal, 0, 5, 4e6, self.myFs/2, 'DownSampler 0')
        evmDown0 = DSPFunctions.evmMeter(down0Signal, symbols, self.myFs/2, 0, False)
        DSPFunctions.signalPlotting(down0Signal, self.myFs/2, f'DownSampler 0. EVM = {evmDown0} dB')

        # First downsampler
        down1Signal = self.Downsampler(down0Signal, 0, 9, 2e6, self.myFs/4, 'DownSampler 1')
        evmDown1 = DSPFunctions.evmMeter(down1Signal, symbols, self.myFs/4, 0, False)
        DSPFunctions.signalPlotting(down1Signal, self.myFs/4, f'DownSampler 1. EVM = {evmDown1} dB')

        # Second downsampler
        down2Signal = self.Downsampler(down1Signal, 0, 9, 1e6, self.myFs/8, 'DownSampler 1')
        evmDown2 = DSPFunctions.evmMeter(down2Signal, symbols, self.myFs/8 , 0, False)
        DSPFunctions.signalPlotting(down2Signal, self.myFs/8, f'DownSampler 2. EVM = {evmDown2} dB')

        DSPFunctions.display()

    def RunOrg(self, signal):
        # FsOver4 Mixer
        fsOver4Signal = self.FsOver4(signal)
        DSPFunctions.signalPlotting(fsOver4Signal, self.myFs, 'RxFsOver4')
        # First downsampler
        down0Signal = self.Downsampler(fsOver4Signal, 0, 11, 4e6, self.myFs/2, 'DownSampler 0')
        DSPFunctions.signalPlotting(down0Signal, self.myFs/2, 'DownSampler 0')
        # Mixing down
        mixDownSignal = self.Mixing(down0Signal, 5e6, self.myFs/2)
        DSPFunctions.signalPlotting(mixDownSignal, self.myFs/2, 'Down Mixer')
        # Second downsampler
        down1Signal = self.Downsampler(mixDownSignal, 0, 5, 6e6, self.myFs/4, 'DownSampler 1')
        DSPFunctions.signalPlotting(down1Signal, self.myFs/4, 'DownSampler 1')
        # Third downsampler
        down2Signal = self.Downsampler(down1Signal, 0, 5, 3e6, self.myFs/8, 'DownSampler 2')
        DSPFunctions.signalPlotting(down2Signal, self.myFs/8, 'DownSampler 2')
        # Fourth downsampler
        down3Signal = self.Downsampler(down2Signal, 0, 5, 1.5e6, self.myFs/16, 'DownSampler 3')
        DSPFunctions.signalPlotting(down3Signal, self.myFs/16, 'DownSampler 3')

        DSPFunctions.display()




