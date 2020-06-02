import transmitter as tx
import receiver as rx

class Simulator:

    def __init__(self):
        self.myNumberOfSymbols = 10
        self.mySeed = 777
        self.myIF = 1.5e6
        self.myModulationScheme = 'QPSK'
        self.mySymbolRate = 1e6;
        self.myTx = tx.Tx(self.myNumberOfSymbols, self.mySeed, self.myModulationScheme, self.myIF, self.mySymbolRate)
        self.myRx = rx.Rx(self.myModulationScheme, self.myIF,  self.myNumberOfSymbols, self.mySymbolRate)

    def Run(self):
        self.myTx.Run()
        self.myRx.Run(self.myTx.GetTxOutput(), self.myTx.GetPacket())
        print('Tx Packet with ', self.myTx.GetPacket().size, ' symbols')
        print(self.myTx.GetPacket())
        print('Rx Packet with ', self.myRx.GetPacket().size, ' symbols')
        print(self.myRx.GetPacket())


mySim = Simulator()
mySim.Run()
