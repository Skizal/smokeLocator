import os
import cv2
import xml.etree.ElementTree as xml


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
    

'''
imagesPath = '/home/enrique/tfm/data/day_time_wildfire_v2_2192/images'
gtPath = '/home/enrique/tfm/data/day_time_wildfire_v2_2192/annotations/xmls'

'''
imagesPath = '/home/enrique/tfm/data/SF_dataset_resized_12620/images'
gtPath = '/home/enrique/tfm/data/SF_dataset_resized_12620/annotations'


#convert( imagesPath, gtPath, 640, 480 )
