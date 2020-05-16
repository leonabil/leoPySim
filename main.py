import transmitter as tx

class Simulator:

    def __init__(self):
        self.myNumberOfSymbols = 200
        self.mySeed = 777
        self.myTx = tx.Tx(self.myNumberOfSymbols, self.mySeed)

    def Run(self):
        self.myTx.Run()

mySim = Simulator()
mySim.Run()
