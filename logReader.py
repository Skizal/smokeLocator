import argparse
import os
import re
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("folder")
args = parser.parse_args()

print( args.folder )

for entry in os.scandir( args.folder ):
    file = open( entry.path, 'r')

    lines = file.readlines()
    losses = []
    vals = []
    for line in lines:
        lossReg = re.search( 'loss: (\d+.\d+)', line )
        if lossReg is not None:
            losses.append( float( lossReg.group().removeprefix("loss: ") ) )
        valReg = re.search( 'val_loss: (\d+.\d+)', line ) 
        if valReg is not None:
            vals.append( float( valReg.group().removeprefix("val_loss: ") ) )

    plt.style.use("ggplot")
    plt.figure()
    plt.plot( range( 0, len(losses) ), losses, label="train_loss" )
    plt.plot( range( 0, len(vals) ), vals, label="val_loss" )
    plt.title( entry.name )
    plt.xlabel( "Epoch #" )
    plt.ylabel( "Loss" )
    plt.legend(loc="lower left")
    plt.show()