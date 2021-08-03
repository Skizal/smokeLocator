import os

import xml.etree.ElementTree as xml

from keras.preprocessing.image import load_img
import numpy as np

from utils import *

def getImageAndBbox( file ):
    root = xml.parse( file ).getroot()
    
    xmin = root.find( './/xmin' )
    ymin = root.find( './/ymin' )
    xmax = root.find( './/xmax' )
    ymax = root.find( './/ymax' )
    name = root.find( './/filename' )
    
    min = Point( float(xmin.text), float(ymin.text) )
    max = Point( float(xmax.text), float(ymax.text) )
    box = BoundingBox( min, max )
    image = ImageData( name.text, '', box, [] )
    
    return image
   

def readAndLoadData( imagesPath, gtPath, imagesToUsePath ):
    imageData = {}

    useFile = open( imagesToUsePath, "r" )
    imagesToUse = useFile.read().splitlines()

    for entry in os.scandir( gtPath ):
        nameNoXml = entry.name.replace( '.xml', '' )
        if nameNoXml in imagesToUse:

            image = getImageAndBbox( entry.path )
            #print( "Image data: ", image.name, image.box.min.x, image.box.min.y, image.box.max.x, image.box.max.y )
            imageData[image.name] = image
        
    for entry in os.scandir( imagesPath ):
        if entry.name in imageData:
            #load the image
            image = imageData[entry.name] 
            iData = load_img( entry.path )
            image.path = entry.path
            image.data = np.asarray( iData )
            image.data = np.swapaxes( image.data, 0, 1)
            imageData[image.name] = image

    return list( imageData.values() )
    
