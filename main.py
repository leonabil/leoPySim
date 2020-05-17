import transmitter as tx
import receiver as rx

class Simulator:

    def __init__(self):
        self.myNumberOfSymbols = 200
        self.mySeed = 777
        self.myModulationScheme = 'QPSK'
        self.myTx = tx.Tx(self.myNumberOfSymbols, self.mySeed, self.myModulationScheme)
        self.myRx = rx.Rx(self.myModulationScheme)

    def Run(self):
        self.myTx.Run()
        self.myRx.Run(self.myTx.GetTxOutput())

mySim = Simulator()
mySim.Run()
