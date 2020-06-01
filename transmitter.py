import numpy as np
import DSPFunctions
from scipy import signal as sgn

class Tx:
    def __init__(self, numberOfSymbols, seed, modulationScheme, IF):
        self.myUpPulseShapeOverSamplingFactor = 8
        self.mySymbolRate = 1e6
        self.mySeed = seed
        self.myNumberOfSymbols = numberOfSymbols
        self.myModulationScheme = modulationScheme
        self.myIF = IF
        self.myPulseTime = np.array([-4.000e-06, -3.875e-06, -3.750e-06, -3.625e-06, -3.500e-06, -3.375e-06, -3.250e-06, -3.125e-06,
                                     -3.000e-06, -2.875e-06, -2.750e-06, -2.625e-06, -2.500e-06, -2.375e-06, -2.250e-06, -2.125e-06,
                                     -2.000e-06, -1.875e-06, -1.750e-06, -1.625e-06, -1.500e-06, -1.375e-06, -1.250e-06, -1.125e-06,
                                     -1.000e-06, -8.750e-07, -7.500e-07, -6.250e-07, -5.000e-07, -3.750e-07, -2.500e-07, -1.250e-07,
                                     0.000e+00,  1.250e-07,  2.500e-07, 3.750e-07,  5.000e-07,  6.250e-07,  7.500e-07,  8.750e-07,
                                     1.000e-06,  1.125e-06,  1.250e-06,  1.375e-06,  1.500e-06,   1.625e-06,  1.750e-06,  1.875e-06,
                                     2.000e-06,  2.125e-06,  2.250e-06,  2.375e-06,  2.500e-06,  2.625e-06,  2.750e-06, 2.875e-06,
                                     3.000e-06,  3.125e-06,  3.250e-06,  3.375e-06,   3.500e-06,  3.625e-06,  3.750e-06,  3.875e-06])
        self.myRrcPulse = np.array([-0.01171875, -0.0078125 , -0.00390625,  0.00390625,  0.01171875, 0.015625  ,  0.015625  ,  0.01171875,
                                    0.00390625, -0.0078125 ,  -0.015625  , -0.01953125, -0.015625  , -0.00390625,  0.015625  , 0.03125   ,
                                    0.04296875,  0.0390625 ,  0.015625  , -0.0234375 ,  -0.07421875, -0.125     , -0.15625   , -0.15625   ,
                                    -0.10546875,-0.        ,  0.15625   ,  0.35546875,  0.578125  ,  0.79296875, 0.97265625,  1.09375   ,
                                    1.13671875,  1.09375   ,  0.97265625,  0.79296875,  0.578125  ,  0.35546875,  0.15625   , -0.        ,
                                    -0.10546875, -0.15625   , -0.15625   , -0.125     , -0.07421875, -0.0234375 ,  0.015625  ,  0.0390625 ,
                                    0.04296875,  0.03125   , 0.015625  , -0.00390625, -0.015625  , -0.01953125, -0.015625  ,  -0.0078125 ,
                                    0.00390625,  0.01171875,  0.015625  ,  0.015625  ,0.01171875,  0.00390625, -0.00390625, -0.0078125 ])

    def GeneratePacket(self):
        np.random.seed(self.mySeed)
        self.mySymbols = np.random.randint(0,4,self.myNumberOfSymbols)
        if self.myModulationScheme == 'QPSK':
            self.mySymbAmp = 1/np.sqrt(2)
            self.myConstellation = np.array([self.mySymbAmp+1j*self.mySymbAmp, self.mySymbAmp-1j*self.mySymbAmp,
                                         -self.mySymbAmp+1j*self.mySymbAmp, -self.mySymbAmp-1j*self.mySymbAmp])
            self.myQPSKSignal = self.myConstellation[self.mySymbols]
        else :
            raise SyntaxError('ERROR: TX::GeneratePacket - Modulation scheme not supported.')

    def Modulator(self, debugMode=False):
        if (debugMode):
                DSPFunctions.timePlot(self.myRrcPulse, self.mySymbolRate * self.myUpPulseShapeOverSamplingFactor, 'Pulse Shaping')
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
        DSPFunctions.freqResponse(upCoeff, fsUp, label, False)
        return outSignal

    def Mixing(self, signal, freqMix, fsMix):
        time = np.arange(0, signal.size) * (1 / fsMix)
        mixedSignal = signal * np.exp(2j * np.pi * freqMix * time)
        return mixedSignal

    def FsOver4(self, signal):
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

        # Packet Generation @ 1Msps
        self.GeneratePacket()

        # Modulation - Pulse shapping at 8 Msps
        pulseShapeSignal = self.Modulator()
        evmMod = DSPFunctions.evmMeter(pulseShapeSignal, self.myQPSKSignal, self.mySymbolRate * self.myUpPulseShapeOverSamplingFactor, 0, False)
        DSPFunctions.signalPlotting(pulseShapeSignal, self.mySymbolRate * self.myUpPulseShapeOverSamplingFactor, f'Pulse Shaping. EVM = {evmMod} dB')


        # First upsampler to 16 Msps.
        fsUp0 = self.mySymbolRate * self.myUpPulseShapeOverSamplingFactor * 2
        up0Signal = self.Upsampling(pulseShapeSignal, 11, 4e6, fsUp0, 'Upsampler 0')
        evmUp0 = DSPFunctions.evmMeter(up0Signal, self.myQPSKSignal, fsUp0, 0, False)
        DSPFunctions.signalPlotting(up0Signal, fsUp0, f'Upsampler 0. EVM = {evmUp0} dB')

        # Second Upsampler to 32 Msps.
        fsUp1 = fsUp0 * 2
        up1Signal = self.Upsampling(up0Signal, 7, 4e6, fsUp1, 'Upsampler 1')
        evmUp1 = DSPFunctions.evmMeter(up1Signal, self.myQPSKSignal, fsUp1, 0, False)
        DSPFunctions.signalPlotting(up1Signal, fsUp1, f'Upsampler 1. EVM = {evmUp1} dB')

        # IFMixer to 1.5 IF.
        self.myTxOutput = self.Mixing(up1Signal, self.myIF, fsUp1)
        evmMixer = DSPFunctions.evmMeter(self.myTxOutput, self.myQPSKSignal, fsUp1, self.myIF, True)
        DSPFunctions.signalPlotting(self.myTxOutput, fsUp1, f'IF Mixer. EVM = {evmMixer} dB')

        DSPFunctions.display()

    def RunOrg(self):

        # Packet Generation
        self.GeneratePacket()
        # Modulation - Pulse shapping
        pulseShapeSignal = self.Modulator()
        #DSPFunctions.signalPlotting(pulseShapeSignal, self.mySymbolRate * self.myUpPulseShapeOverSamplingFactor, 'Pulse Shaping')
        print('EVM Pulse Shapping: ', DSPFunctions.evmMeter(pulseShapeSignal, self.myQPSKSignal, self.mySymbolRate * self.myUpPulseShapeOverSamplingFactor, 0, False))
        # First upsampler
        fsUp0 = self.mySymbolRate * self.myUpPulseShapeOverSamplingFactor * 2
        up0Signal = self.Upsampling(pulseShapeSignal, 9, 2e6, fsUp0, 'Upsampler 0')
        #DSPFunctions.signalPlotting(up0Signal, fsUp0, 'Upsampler 0')
        print('EVM fsUp0: ', DSPFunctions.evmMeter(up0Signal, self.myQPSKSignal, fsUp0, 0, False))
        # Second Upsampler
        fsUp1 = fsUp0 * 2
        up1Signal = self.Upsampling(up0Signal, 9, 2e6, fsUp1, 'Upsampler 1')
        #DSPFunctions.signalPlotting(up1Signal, fsUp1, 'Upsampler 1')
        print('EVM fsUp1: ', DSPFunctions.evmMeter(up1Signal, self.myQPSKSignal, fsUp1, 0, False))
        # Third upsampler
        fsUp2 = fsUp1 * 2
        up2Signal = self.Upsampling(up1Signal, 9, 2e6, fsUp2, 'Upsampler 2')
        #DSPFunctions.signalPlotting(up2Signal, fsUp2, 'Upsampler 2')
        print('EVM fsUp2: ', DSPFunctions.evmMeter(up2Signal, self.myQPSKSignal, fsUp2, 0, False))
        # Mixer
        mixedSignal = self.Mixing(up2Signal, 5e6, fsUp2)
        #DSPFunctions.signalPlotting(mixedSignal, fsUp2, 'Channel Mixer')
        print('EVM Mixer: ', DSPFunctions.evmMeter(mixedSignal, self.myQPSKSignal, fsUp2, 5e6, True))
        # Fourth upsamler
        fsUp3 = fsUp2 * 2
        up3Signal = self.Upsampling(mixedSignal, 11, 12e6, fsUp3, 'Upsampler 3')
        #DSPFunctions.signalPlotting(up3Signal, fsUp3, 'Upsampler 3')
        print('EVM fsUp3: ', DSPFunctions.evmMeter(up3Signal, self.myQPSKSignal, fsUp3, 5e6))
        # FsOver4 Mixer
        self.myTxOutput = self.FsOver4(up3Signal)
        #DSPFunctions.signalPlotting(self.myTxOutput, fsUp3, 'FsOver4')
        print('EVM FsOver4: ', DSPFunctions.evmMeter(self.myTxOutput, self.myQPSKSignal, fsUp3, 21e6))
        #DSPFunctions.display()

    def GetTxOutput(self):
        return self.myTxOutput

    def GetPacket(self):
        return self.myQPSKSignal


