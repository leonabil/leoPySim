import numpy as np
import DSPFunctions
from scipy import signal as sgn

class Tx:
    def __init__(self, numberOfSymbols, seed):
        self.myUpPulseShapeOverSamplingFactor = 4
        self.mySamplingFreq = 1e6
        self.mySeed = seed
        self.myNumberOfSymbols = numberOfSymbols
        self.myPulseTime = np.array([-4.00e-06, -3.75e-06, -3.50e-06, -3.25e-06, -3.00e-06, -2.75e-06,
                                     -2.50e-06, -2.25e-06, -2.00e-06, -1.75e-06, -1.50e-06, -1.25e-06,
                                     -1.00e-06, -7.50e-07, -5.00e-07, -2.50e-07, 0.00e+00, 2.50e-07,
                                     5.00e-07, 7.50e-07, 1.00e-06, 1.25e-06, 1.50e-06, 1.75e-06,
                                     2.00e-06, 2.25e-06, 2.50e-06, 2.75e-06, 3.00e-06, 3.25e-06,
                                     3.50e-06, 3.75e-06])
        self.myRrcPulse = np.array([-0.00583243, 0.00857444, 0.0150285, 0.00364796, -0.01586493,
                                    -0.01909873, 0.00848826, 0.0484814, 0.05471622, -0.00748792,
                                    -0.11553457, -0.18006326, -0.09317284, 0.19053998, 0.59911012,
                                    0.96339776, 1.10929582, 0.96339776, 0.59911012, 0.19053998,
                                    -0.09317284, -0.18006326, -0.11553457, -0.00748792, 0.05471622,
                                    0.0484814, 0.00848826, -0.01909873, -0.01586493, 0.00364796,
                                    0.0150285, 0.00857444])

    def GeneratePacket(self):
        np.random.seed(self.mySeed)
        self.mySymbols = np.random.randint(0,4,self.myNumberOfSymbols)
        self.mySymbAmp = 1/np.sqrt(2)
        self.myConstellation = np.array([self.mySymbAmp+1j*self.mySymbAmp, self.mySymbAmp-1j*self.mySymbAmp,
                                         -self.mySymbAmp+1j*self.mySymbAmp, -self.mySymbAmp-1j*self.mySymbAmp])
        self.myQPSKSignal = self.myConstellation[self.mySymbols]

    def Modulator(self):
        DSPFunctions.timePlot(self.myRrcPulse, self.mySamplingFreq, 'Pulse Shaping')
        upTemp = np.zeros((self.myQPSKSignal.size, self.myUpPulseShapeOverSamplingFactor)) \
                 + 1j * np.zeros((self.myQPSKSignal.size, self.myUpPulseShapeOverSamplingFactor))
        upTemp[:, 0] = self.myQPSKSignal
        upSignal = np.reshape(upTemp, (1, self.myQPSKSignal.size * self.myUpPulseShapeOverSamplingFactor))
        pulseShapeSignal = np.convolve(upSignal[0], self.myRrcPulse)
        return pulseShapeSignal

    def Upsampling(self, signal, order, flatBand, fsUp, label):
        upSamplingFactorUp = 2
        upSignal = np.zeros(signal.size * upSamplingFactorUp) + 1j * np.zeros(signal.size * upSamplingFactorUp)
        for counter1 in range(0, signal.size - 1):
            upSignal[2 * counter1] = signal[counter1]
        upCoeff = sgn.firwin(order+1, flatBand, fs=fsUp)
        outSignal = np.convolve(upSignal, upCoeff)
        # DSPFunctions.freqResponse(upCoeff, fsUp, label)
        return outSignal

    def Mixing(self, signal, freqMix, fsMix):
        time = np.arange(0, signal.size) * (1 / fsMix)
        mixedSignal = signal * np.exp(2j * np.pi * freqMix * time)
        return mixedSignal

    def FsOver4(self, signal, fs):
        switcher = {
            0: np.exp(1j * np.pi * 0 / 2),
            1: np.exp(1j * np.pi * 1 / 2),
            2: np.exp(1j * np.pi * 2 / 2),
            3: np.exp(1j * np.pi * 3 / 2),
        }
        fsOver4Signal = np.zeros(signal.size) + 1j * np.zeros(signal.size)
        for counter1 in range(0, signal.size - 1):
            fsOver4Signal[counter1] = signal[counter1] * switcher.get(counter1 % 4)
        return fsOver4Signal

    def Run(self):
        self.GeneratePacket()
        pulseShapeSignal = self.Modulator()
        DSPFunctions.signalPlotting(pulseShapeSignal, self.mySamplingFreq * self.myUpPulseShapeOverSamplingFactor, 'Pulse Shaping')
        fsUp0 = self.mySamplingFreq * self.myUpPulseShapeOverSamplingFactor * 2
        up0Signal = self.Upsampling(pulseShapeSignal, 9, 2e6, fsUp0, 'Upsampler 0')
        DSPFunctions.signalPlotting(up0Signal, fsUp0, 'Upsampler 0')
        fsUp1 = fsUp0 * 2
        up1Signal = self.Upsampling(up0Signal, 8, 2e6, fsUp1, 'Upsampler 1')
        DSPFunctions.signalPlotting(up1Signal, fsUp1, 'Upsampler 1')
        fsUp2 = fsUp1 * 2
        up2Signal = self.Upsampling(up1Signal, 5, 2e6, fsUp2, 'Upsampler 2')
        DSPFunctions.signalPlotting(up2Signal, fsUp2, 'Upsampler 2')
        mixedSignal = self.Mixing(up2Signal, 5e6, fsUp2)
        DSPFunctions.signalPlotting(mixedSignal, fsUp2, 'Channel Mixer')
        fsUp3 = fsUp2 * 2
        up3Signal = self.Upsampling(mixedSignal, 11, 12e6, fsUp3, 'Upsampler 3')
        DSPFunctions.signalPlotting(up3Signal, fsUp3, 'Upsampler 3')
        self.myTxOutput = self.FsOver4(up3Signal, fsUp3)
        DSPFunctions.signalPlotting(self.myTxOutput, fsUp3, 'FsOver4')
        DSPFunctions.display()

    def GetTxOutout(self):
        return self.myTxOutput



