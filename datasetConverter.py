import os
import cv2
import xml.etree.ElementTree as xml 
from utils import Configuration
import random


#bbox is changed from absolute to relative size. 
def changeBboxFromXML( file, resX, resY ) :

    tree = xml.parse( file )
    root = tree.getroot()
    
    exmin = root.find( './/xmin' )
    eymin = root.find( './/ymin' )
    exmax = root.find( './/xmax' )
    eymax = root.find( './/ymax' )

    xmin = float( exmin.text )
    ymin = float( eymin.text )
    xmax = float( exmax.text )
    ymax = float( eymax.text )

    #This is to avoid converting already relative boxes
    if xmax > 1:
        # Resize ground truth box 
        newXmin = xmin / resX
        newYmin = ymin / resY
        newXmax = xmax / resX  
        newYmax = ymax / resY  

        print( 'Old bbox:', xmin, ymin, xmax, ymax )
        print( 'New bbox', newXmin, newYmin, newXmax, newYmax )

        exmin.text = str(newXmin)
        eymin.text = str(newYmin)
        exmax.text = str(newXmax)
        eymax.text = str(newYmax)

        tree.write( file )
        


def convert( imagesPath, gtPath, newResX, newResY ):
    for entry in os.scandir( gtPath ):
        print(entry.name)
        changeBboxFromXML( entry.path, newResX, newResY )

    
    for entry in os.scandir( imagesPath ):
        print(entry.name)
        image = cv2.imread( entry.path )
        newImage = cv2.resize( image, ( newResX, newResY ) )
        cv2.imwrite( entry.path, newImage )
    


def updateImagesToUse( gtPath, usePath, testPath, nImages ):

    list = os.listdir( gtPath )
    random.shuffle( list )

    toUse = list[:nImages]
    textFile = open( usePath , "w" )
    for elem in toUse:
        e = elem.replace( '.xml', '' )
        textFile.write( e + "\n" )
    textFile.close()

    toTest = list[nImages:]
    textFile = open( testPath , "w" )
    for elem in toTest:
        e = elem.replace( '.xml', '' )
        textFile.write( e + "\n" )
    textFile.close()


updateImagesToUse( Configuration.trainGT[0], Configuration.trainUsage[0], Configuration.testUsage[0], 1800 )
updateImagesToUse( Configuration.trainGT[1], Configuration.trainUsage[1], Configuration.testUsage[1], 1800 )

#convert( Configuration.testImages, Configuration.testGT, 480, 360 )
