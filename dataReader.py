import os
import random
import cv2
import xml.etree.ElementTree as xml


from utils import ImageData, Point

def getImageAndBbox( file ):
    root = xml.parse( file ).getroot()
    
    xmin = root.find( './/xmin' )
    ymin = root.find( './/ymin' )
    xmax = root.find( './/xmax' )
    ymax = root.find( './/ymax' )
    name = root.find( './/filename' )
    
    pointA = Point( xmin.text, ymin.text )
    pointB = Point( xmax.text, ymax.text )
    image = ImageData( name.text, pointA, pointB, [] )
    
    return image
   

def readAndLoadData( imagesPath, gtPath, imagesToUsePath ):
    imageData = {}

    useFile = open( imagesToUsePath, "r" )
    imagesToUse = useFile.read().splitlines()

    for entry in os.scandir( gtPath ):
        nameNoXml = entry.name.replace( '.xml', '' )
        if nameNoXml in imagesToUse:

            image = getImageAndBbox( entry.path )
            print( "Image data: ", image.name, image.pointA.x, image.pointA.y, image.pointB.x, image.pointB.y )
            imageData[image.name] = image
        
    for entry in os.scandir( imagesPath ):
        if entry.name in imageData:
            #load the image
            iData = cv2.imread( entry.path )
            image = imageData[entry.name] 
            image.data = iData
            imageData[image.name] = image

    return list( imageData.values() )
     

def updateImagesToUse( gtPath, usePath, nImages ):

    list = os.listdir( gtPath )
    random.shuffle( list )
    textFile = open( usePath , "w" )
    for elem in list[:nImages]:
        e = elem.replace( '.xml', '' )
        textFile.write( e + "\n" )
    textFile.close()

'''
testImagesPath = '/home/enrique/tfm/data/day_time_wildfire_v2_2192/images'
testGTPath = '/home/enrique/tfm/data/day_time_wildfire_v2_2192/annotations/xmls'

trainImagesPath = '/home/enrique/tfm/data/SF_dataset_resized_12620/images'
trainGTPath = '/home/enrique/tfm/data/SF_dataset_resized_12620/annotations'

usePath1 = "/home/enrique/tfm/data/SF_dataset_resized_12620/usedImages.txt"
usePath2 = "/home/enrique/tfm/data/day_time_wildfire_v2_2192/usedImages.txt"

updateImagesToUse( trainGTPath, usePath1, 1000)
updateImagesToUse( testGTPath, usePath2, 250 )
'''

#trainData = readAndLoadData( trainImagesPath, trainGTPath, usePath1 )