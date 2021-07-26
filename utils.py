#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 22:26:11 2021

@author: enrique
"""

from dataclasses import dataclass
from typing import List

@dataclass
class Point:
     x: float
     y: float

@dataclass
class BoundingBox:
     min: Point
     max: Point

@dataclass
class ImageData:
    name: str
    path: str
    box: BoundingBox
    data: List[int]

@dataclass
class Configuration:
     trainImages: str = '/home/enrique/tfm/data/SF_dataset_resized_12620/images'
     trainGT: str = '/home/enrique/tfm/data/SF_dataset_resized_12620/annotations'
     trainUsage: str = '/home/enrique/tfm/data/SF_dataset_resized_12620/usedImages.txt'
     testImages: str = '/home/enrique/tfm/data/day_time_wildfire_v2_2192/images'
     testGT: str = '/home/enrique/tfm/data/day_time_wildfire_v2_2192/annotations/xmls'
     testUsage: str = '/home/enrique/tfm/data/day_time_wildfire_v2_2192/usedImages.txt'
     output: str = '/home/enrique/tfm/output/'

@dataclass
class Detection:
     imagePath: str
     gt: BoundingBox
     pred: BoundingBox



def getBoxesWithAbsoluteIntegerValues( boxGT, boxP, resX, resY ):
    #convert boxes from relative to absolute values
    boxGT.min.x = int( boxGT.min.x * resX )
    boxGT.min.y = int( boxGT.min.y * resY )
    boxGT.max.x = int( boxGT.max.x * resX )
    boxGT.max.y = int( boxGT.max.y * resY )
    
    boxP.min.x = int( boxP.min.x * resX )
    boxP.min.y = int( boxP.min.y * resY )
    boxP.max.x = int( boxP.max.x * resX )
    boxP.max.y = int( boxP.max.y * resY )

    return boxGT, boxP


def intersectionOverUnion( boxGT, boxP ):

    #determine coordinates of the intersection rectangle
    xA = max( boxGT.min.x, boxP.min.x )
    yA = max( boxGT.min.y, boxP.min.y )
    xB = min( boxGT.max.x, boxP.max.x )
    yB = min( boxGT.max.y, boxP.max.y )
    
    #compute area of intersection rectangle
    interArea = max( 0, xB - xA + 1) * max( 0, yB - yA + 1 )
    
    #compute area of Bbox union
    areaGT = ( boxGT.max.x - boxGT.min.x + 1 ) * ( boxGT.max.y - boxGT.min.y + 1 )
    areaP = ( boxP.max.x - boxP.min.x + 1 ) * ( boxP.max.y - boxP.min.y + 1 )
    
    unionArea = areaGT + areaP - interArea
    
    iou = interArea / unionArea
    
    return iou



    