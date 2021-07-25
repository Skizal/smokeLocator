
from tensorflow.keras.models import load_model
import numpy as np

import dataReader as reader
from utils import ImageData, Point

modelPath = "/home/enrique/tfm/output/model"
testImagesPath = '/home/enrique/tfm/data/day_time_wildfire_v2_2192/images'
testGTPath = '/home/enrique/tfm/data/day_time_wildfire_v2_2192/annotations/xmls'
testUsePath = "/home/enrique/tfm/data/day_time_wildfire_v2_2192/usedImages.txt"

testData = reader.readAndLoadData( testImagesPath, testGTPath, testUsePath )
testImages = [ img.data for img in testData ]
testBbox = [ [img.pointA.x, img.pointA.y, img.pointB.x, img.pointB.y] for img in testData ]

testImages = np.array( testImages, dtype = "float32") / 255.0 

model = load_model( modelPath )
image = []
image.append( np.array( testImages[0] ) )
x = np.asarray( image ) 
preds = model.predict( x )[0]
(startX, startY, endX, endY) = preds

print( preds )