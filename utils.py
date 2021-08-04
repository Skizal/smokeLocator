
from dataclasses import dataclass, field
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
     trainImages = [ '/home/enrique/tfm/data/SF_dataset_resized_12620/images', '/home/enrique/tfm/data/day_time_wildfire_v2_2192/images' ]
     trainGT = [ '/home/enrique/tfm/data/SF_dataset_resized_12620/annotations', '/home/enrique/tfm/data/day_time_wildfire_v2_2192/annotations/xmls' ] 
     trainUsage = [ '/home/enrique/tfm/data/SF_dataset_resized_12620/usedImages.txt', '/home/enrique/tfm/data/day_time_wildfire_v2_2192/usedImages.txt' ]
     testUsage = [ '/home/enrique/tfm/data/SF_dataset_resized_12620/testImages.txt', '/home/enrique/tfm/data/day_time_wildfire_v2_2192/testImages.txt' ]
     modelPath = '/home/enrique/tfm/output/'

     batchSize = [1, 8, 16, 32] 
     nEpochs = 25

     learningRate = [1e-4, 1e-5]

     limitImages = 1800

     xRes = 480
     yRes = 360

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
    
