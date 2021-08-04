
from tensorflow.keras.models import load_model
from keras.optimizers import Adam
import numpy as np

import dataReader as reader
from collections import namedtuple
from utils import *
from losses import *

import cv2

def predictAndShowImages():

    
    pred= model.predict( testImages )
    eval = model.evaluate( testImages, testBbox )
    print( pred )
    print( eval )

    for index, val in enumerate(testImages):
        min = Point( pred[index][0], pred[index][1] )        
        max = Point( pred[index][2], pred[index][3] )
        predBox = BoundingBox( min, max )

        boxGT, boxP = getBoxesWithAbsoluteIntegerValues( testData[index].box, predBox, Configuration.xRes, Configuration.yRes )

        detection = Detection( testData[index].path, boxGT, boxP )
            
        #load the image
        image = cv2.imread( detection.imagePath )
        
        #draw the ground-truth bbox along with the predicted bbox (BGR)
        cv2.rectangle( image, ( detection.gt.min.x, detection.gt.min.y ), ( detection.gt.max.x, detection.gt.max.y ), (0, 255, 0 ), 2 )
        cv2.rectangle( image, ( detection.pred.min.x, detection.pred.min.y ), ( detection.pred.max.x, detection.pred.max.y ), (0, 0, 255 ), 2 )
        
        #compute iou and display it
        (exA, eyA, exB, eyB, centerGTX, centerGTY, centerPX, centerPY ) = distanceIoUinfo( detection.gt, detection.pred )

        print( "Ground truth: [" + str( detection.gt.min.x ) + "," + str( detection.gt.min.y ) + "," + str( detection.gt.max.x ) + "," + str( detection.gt.max.y ) + "]" )
        print( "Prediction: [" + str( detection.pred.min.x ) + "," + str( detection.pred.min.y ) + "," + str( detection.pred.max.x ) + "," + str( detection.pred.max.y ) + "]" )
        
        cv2.line( image, (exA, eyA), (exB, eyB), (139, 0, 139), 2 )
        cv2.line( image, (centerGTX, centerGTY), (centerPX, centerPY), (230, 230, 250), 2 )
        #show output image
        cv2.imshow( "Image", image )
        cv2.waitKey(0)


nImages = 10

data = reader.readAndLoadData( Configuration.trainImages[0], Configuration.trainGT[0], Configuration.testUsage[0], Configuration.limitImages )

#random.shuffle( data )
testData = data[:nImages]

testImages = [ img.data for img in testData ]
testBbox = [ [ img.box.min.x, img.box.min.y, img.box.max.x, img.box.max.y ] for img in testData ]

testImages = np.array( testImages, dtype = "float32") / 255.0 
testBbox = np.array( testBbox, dtype = "float32")


model = load_model( Configuration.modelPath + "/25epochs/model_" + str(0) + "_" + str(Configuration.batchSize[2]) + "_" + str(Configuration.learningRate[0]), compile=False )
opt = Adam( lr = Configuration.learningRate[0] )
model.compile( optimizer = opt, loss = [diouCoef] )

predictAndShowImages()