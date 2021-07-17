#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 12:38:13 2021

@author: enrique
"""

import os
import cv2
import xml.etree.ElementTree as xml
from utils import ImageData, Point

def getImageAndBbox( file ):
    root = xml.parse( file ).getroot()
    
    xmin= root.find( './/xmin' )
    ymin = root.find( './/ymin' )
    xmax = root.find( './/xmax' )
    ymax = root.find( './/ymax' )
    name = root.find( './/filename' )
    
    pointA = Point( xmin.text, ymin.text )
    pointB = Point( xmax.text, ymax.text )
    image = ImageData( name.text, pointA, pointB, [] )
    
    return image
   

def readAndLoadData( imagesPath, gtPath ):
    imageData = {}

    with os.scandir( '/media/enrique/Dades/backup/TFMData/day_time_wildfire_v2_2192/annotations/xmls' ) as entries:
        for entry in entries:
            print(entry.name)
            image = getImageAndBbox( entry.path )
            print( "Image data: ", image.name, image.pointA.x, image.pointA.y, image.pointB.x, image.pointB.y )
            imageData[image.name] = image
        
    with os.scandir( '/media/enrique/Dades/backup/TFMData/day_time_wildfire_v2_2192/images' ) as entries:
        for entry in entries:
            print(entry.name)
            #load the image
            iData = cv2.imread( entry.path )
            image = imageData[entry.name] 
            image.data = iData
            imageData[image.name] = image
    
    return imageData


imagesPath = '/media/enrique/Dades/backup/TFMData/day_time_wildfire_v2_2192/images'
gtPath = '/media/enrique/Dades/backup/TFMData/day_time_wildfire_v2_2192/annotations/xmls'

trainData = readAndLoadData( imagesPath, gtPath )