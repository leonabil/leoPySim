import transmitter as tx
import receiver as rx

class Simulator:

    def __init__(self):
        self.myNumberOfSymbols = 200
        self.mySeed = 777
        self.myIF = 1.5e6
        self.myModulationScheme = 'QPSK'
        self.myTx = tx.Tx(self.myNumberOfSymbols, self.mySeed, self.myModulationScheme, self.myIF)
        self.myRx = rx.Rx(self.myModulationScheme, self.myIF)

    def Run(self):
        self.myTx.Run()
        self.myRx.Run(self.myTx.GetTxOutput(), self.myTx.GetPacket())

mySim = Simulator()
mySim.Run()
