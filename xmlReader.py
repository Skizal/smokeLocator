#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 12:38:13 2021

@author: enrique
"""

import xml.etree.ElementTree as xml
from utils import ImageMetadata, Point

def getImageAndBbox( file ):
    root = xml.parse( file ).getroot()
    
    xmin= root.find( './/xmin' )
    ymin = root.find( './/ymin' )
    xmax = root.find( './/xmax' )
    ymax = root.find( './/ymax' )
    name = root.find( './/filename' )
    
    pointA = Point( xmin.text, ymin.text )
    pointB = Point( xmax.text, ymax.text )
    image = ImageMetadata( name.text, pointA, pointB )
    
    return image
   

image = getImageAndBbox( "smoke.xml" )

print( "Image data: ", image.name, image.pointA.x, image.pointA.y, image.pointB.x, image.pointB.y )

#Now we need to save all xml files data in a structure and read images linked to them