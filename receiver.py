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

    def Run(self, signal):
        # FsOver4 Mixer
        fsOver4Signal = self.FsOver4(signal)
        DSPFunctions.signalPlotting(fsOver4Signal, self.fs, 'RxFsOver4')
        DSPFunctions.display()




