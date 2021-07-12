#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 20:38:26 2021

@author: enrique
"""

from PIL import Image
from numpy import asarray

image = Image.open("smoke.jpeg")

print( image.format )
print( image.mode )
print( image.size )

data = asarray( image )

print( data.shape )

image2 = Image.fromarray( data )

print( image2.format )
print( image2.mode )
print( image2.size )

image2.save( 'smoke2.jpeg' )
image2.show()
