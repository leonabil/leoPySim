import numpy as np
import DSPFunctions
from scipy import signal as sgn

class Rx:
    # Constructor
    def __init__(self, modulationScheme):
        self.fs = 64e6;
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

    def Downsampler(self, signal, downsamplinPhase, order, flatBand, fsDown, label):
        downSamplingFactor = 2
        downCoeff = sgn.firwin(order + 1, flatBand, fs=fsDown)
        DSPFunctions.freqResponse(downCoeff, fsDown, label)
        filteredSignal = np.convolve(signal, downCoeff)
        outSignal = filteredSignal[downsamplinPhase::downSamplingFactor]
        return outSignal

    def Mixing(self, signal, freqMix, fsMix):
        phase = 2 * np.pi * np.ones(signal.size) * freqMix / fsMix
        mixedSignal = signal * np.exp(-1j * phase.cumsum())
        return mixedSignal

    def Run(self, signal):
        # FsOver4 Mixer
        fsOver4Signal = self.FsOver4(signal)
        DSPFunctions.signalPlotting(fsOver4Signal, self.fs, 'RxFsOver4')
        # First downsampler
        down0Signal = self.Downsampler(fsOver4Signal, 0, 11, 12e6, self.fs/2, 'DownSampler 0')
        DSPFunctions.signalPlotting(down0Signal, self.fs/2, 'DownSampler 0')
        # Mixing down
        mixDownSignal = self.Mixing(down0Signal, 5e6, self.fs/2)
        DSPFunctions.signalPlotting(mixDownSignal, self.fs/2, 'Down Mixer')
        # Second downsampler
        down1Signal = self.Downsampler(mixDownSignal, 0, 5, 6e6, self.fs/4, 'DownSampler 1')
        DSPFunctions.signalPlotting(down1Signal, self.fs/4, 'DownSampler 1')
        # Third downsampler
        down2Signal = self.Downsampler(down1Signal, 0, 5, 3e6, self.fs/8, 'DownSampler 2')
        DSPFunctions.signalPlotting(down2Signal, self.fs/8, 'DownSampler 2')
        # Fourth downsampler
        down3Signal = self.Downsampler(down2Signal, 0, 5, 1.5e6, self.fs/16, 'DownSampler 3')
        DSPFunctions.signalPlotting(down3Signal, self.fs/16, 'DownSampler 3')

        DSPFunctions.display()




