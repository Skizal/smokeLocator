
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
     modelPath: str = '/home/enrique/tfm/output/'
     plotPath: str = '/home/enrique/tfm/output/plot/'

     batchSize: int = 10
     nEpochs: int = 30

     learningRate: float = 1e-4

     xRes: int = 480
     yRes: int = 360

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
    

