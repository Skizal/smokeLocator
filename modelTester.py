
from tensorflow.keras.models import load_model
from keras.optimizers import Adam
import numpy as np

import dataReader as reader
from collections import namedtuple
from utils import *
import cv2

nImages = 10
xRes = 640
yRes = 480

data = reader.readAndLoadData( Configuration.trainImages, Configuration.trainGT, Configuration.trainUsage )

testData = data[:nImages]

testImages = [ img.data for img in testData ]
testImages = np.array( testImages, dtype = "float32") / 255.0 

model = load_model( Configuration.modelPath + 'model', compile=False )
opt = Adam( lr = Configuration.learningRate )
model.compile( optimizer = opt, loss = [ciouCoef] )

predictions = model.predict( testImages )

print( predictions )

sample = []
for index, pred in enumerate( predictions ):
    min = Point( pred[0], pred[1] )
    max = Point( pred[2], pred[3] )
    predBox = BoundingBox( min, max )

    boxGT, boxP = getBoxesWithAbsoluteIntegerValues( testData[index].box, predBox, xRes, yRes )

    sample.append( Detection( testData[index].path, boxGT, boxP ) ) 


for detection in sample:
    
     #load the image
     image = cv2.imread( detection.imagePath )
     
     #draw the ground-truth bbox along with the predicted bbox (BGR)
     cv2.rectangle( image, ( detection.gt.min.x, detection.gt.min.y ), ( detection.gt.max.x, detection.gt.max.y ), (0, 255, 0 ), 2 )
     cv2.rectangle( image, ( detection.pred.min.x, detection.pred.min.y ), ( detection.pred.max.x, detection.pred.max.y ), (0, 0, 255 ), 2 )
     
     #compute iou and display it
     iou = completeIou( boxGT, boxP )
     cv2.putText( image, "IoU: {:.4f}".format(iou), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ( 0, 255, 0 ), 2 )

     print( "Ground truth: [" + str( detection.gt.min.x ) + "," + str( detection.gt.min.y ) + "," + str( detection.gt.max.x ) + "," + str( detection.gt.max.y ) + "]" )
     print( "Prediction: [" + str( detection.pred.min.x ) + "," + str( detection.pred.min.y ) + "," + str( detection.pred.max.x ) + "," + str( detection.pred.max.y ) + "]" )
     
     #show output image
     cv2.imshow( "Image", image )
     cv2.waitKey(0)