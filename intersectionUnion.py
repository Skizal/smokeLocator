#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 19:52:38 2021

@author: enrique
"""

from collections import namedtuple
import numpy as np
import cv2

Detection = namedtuple( "Detection", ["image_path", "gt", "pred"] ) 


def intersectionOverUnion( boxGT, boxP ):
    
    #determine coordinates of the intersection rectangle
    xA = max( boxGT[0], boxP[0] )
    yA = max( boxGT[1], boxP[1] )
    xB = min( boxGT[2], boxP[2] )
    yB = min( boxGT[3], boxP[3] )
    
    #compute area of intersection rectangle
    interArea = max( 0, xB - xA + 1) * max( 0, yB - yA + 1 )
    
    #compute area of Bbox union
    areaGT = ( boxGT[2] - boxGT[0] + 1 ) * ( boxGT[3] - boxGT[1] + 1 )
    areaP = ( boxP[2] - boxP[0] + 1 ) * ( boxP[3] - boxP[1] + 1 )
    
    unionArea = areaGT + areaP - interArea
    
    iou = interArea / unionArea
    
    return iou


sample = [ Detection( "smoke.jpeg", [549, 169, 639, 275], [500, 100, 550, 200] ) ]

for detection in sample:
    
     #load the image
     image = cv2.imread( detection.image_path )
     
     #draw the ground-truth bbox along with the predicted bbox
     cv2.rectangle( image, tuple( detection.gt[:2] ), tuple( detection.gt[2:] ), (0, 255, 0 ), 2)
     cv2.rectangle( image, tuple( detection.pred[:2] ), tuple( detection.pred[2:] ), (0, 0, 255 ), 2)
     
     #compute iou and display it
     iou = intersectionOverUnion( detection.gt, detection.pred )
     cv2.putText( image, "IoU: {:.4f}".format(iou), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ( 0, 255, 0 ), 2 )
     
     print("{}: {:.4f}".format( detection.image_path, iou ) )
     
     #show output image
     cv2.imshow( "Image", image )
     cv2.waitKey(0)