
from dataclasses import dataclass
from typing import List
import tensorflow as tf
import math

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
     modelPath: str = '/home/enrique/tfm/output/'
     plotPath: str = '/home/enrique/tfm/output/plot/'

     batchSize: int = 10
     nEpochs: int = 5

     learningRate: float = 1e-4

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



def completeIou( boxGT, boxP ):

    #determine coordinates of the intersection rectangle
    xA = max( boxGT.min.x, boxP.min.x )
    yA = max( boxGT.min.y, boxP.min.y )
    xB = min( boxGT.max.x, boxP.max.x )
    yB = min( boxGT.max.y, boxP.max.y )

     #compute area of intersection rectangle
    interArea = max( 0, xB - xA ) * max( 0, yB - yA )
    
    #compute area of Bbox union
    areaGT = ( boxGT.max.x - boxGT.min.x ) * ( boxGT.max.y - boxGT.min.y )
    areaP = ( boxP.max.x - boxP.min.x ) * ( boxP.max.y - boxP.min.y )
    
    unionArea = areaGT + areaP - interArea
    
    iou = interArea / unionArea

    #Get diagonal between box centers

    centerGTX = ( boxGT.min.x + boxGT.max.x ) * 0.5
    centerGTY = ( boxGT.min.y + boxGT.max.y ) * 0.5
    centerPX = ( boxP.min.x + boxP.max.x ) * 0.5
    centerPY = ( boxP.min.y + boxP.max.y ) * 0.5

    distX = abs( centerPX - centerGTX )
    distY = abs( centerPY - centerGTY )


    #Get diagonal for enclosing box
    exA = min( boxGT.min.x, boxP.min.x ) * 1.0
    eyA = min( boxGT.min.y, boxP.min.y ) * 1.0
    exB = max( boxGT.max.x, boxP.max.x ) * 1.0
    eyB = max( boxGT.max.y, boxP.max.y ) * 1.0

    eDistX = abs( exA - exB )
    eDistY = abs( eyA - eyB )

    centerDis = math.sqrt( distX**2 + distY**2 )
    encloseDis = math.sqrt( eDistX**2 + eDistY**2 )

    diou = iou - ( centerDis**2 / encloseDis**2 )

    print( "Diou: " + str(diou) + " centerDistance: " + str(centerDis) + " encloseDis: " + str(encloseDis) + " iou: " + str(iou) )
    return ( int(exA), int(eyA), int(exB), int(eyB), int(centerGTX), int(centerGTY), int(centerPX), int(centerPY) )
    


def ciouCoef( boxGT, boxP ):

     center_vec = abs( ( boxGT[..., :2] + boxGT[..., 2:] ) * 0.5 - ( boxP[..., :2] + boxP[..., 2:] ) * 0.5 )

     boxGT = tf.concat([boxGT[..., :2] - boxGT[..., 2:] * 0.5,
                         boxGT[..., :2] + boxGT[..., 2:] * 0.5], axis=-1)
     boxP = tf.concat([boxP[..., :2] - boxP[..., 2:] * 0.5,
                         boxP[..., :2] + boxP[..., 2:] * 0.5], axis=-1)

     boxGT = tf.concat([tf.minimum(boxGT[..., :2], boxGT[..., 2:]),
                         tf.maximum(boxGT[..., :2], boxGT[..., 2:])], axis=-1)
     boxP = tf.concat([tf.minimum(boxP[..., :2], boxP[..., 2:]),
                         tf.maximum(boxP[..., :2], boxP[..., 2:])], axis=-1)

     boxes1_area = (boxGT[..., 2] - boxGT[..., 0]) * (boxGT[..., 3] - boxGT[..., 1])
     boxes2_area = (boxP[..., 2] - boxP[..., 0]) * (boxP[..., 3] - boxP[..., 1])

     left_up = tf.maximum(boxGT[..., :2], boxP[..., :2])
     right_down = tf.minimum(boxGT[..., 2:], boxP[..., 2:])

     inter_section = tf.maximum(right_down - left_up, 0.0)
     inter_area = inter_section[..., 0] * inter_section[..., 1]
     union_area = boxes1_area + boxes2_area - inter_area
     iou = inter_area / union_area

     enclose_left_up = tf.minimum(boxGT[..., :2], boxP[..., :2])
     enclose_right_down = tf.maximum(boxGT[..., 2:], boxP[..., 2:])
     enclose_vec = enclose_right_down - enclose_left_up
     center_dis = tf.sqrt( center_vec[..., 0]**2 + center_vec[..., 1]**2 )
     enclose_dis = tf.sqrt( enclose_vec[..., 0]**2 + enclose_vec[..., 1]**2 )

     diou = iou - ( center_dis**2 / enclose_dis**2 )

     return -diou

    