from tensorflow import keras

from keras.applications import VGG16
from keras.models import Model
from keras.layers import Dense, Flatten, Input
from keras.optimizers import Adam
import numpy as np

import dataReader as reader
from utils import ImageData, Point

outPath = "/home/enrique/tfm/output/"

testImagesPath = '/home/enrique/tfm/data/day_time_wildfire_v2_2192/images'
testGTPath = '/home/enrique/tfm/data/day_time_wildfire_v2_2192/annotations/xmls'
testUsePath = "/home/enrique/tfm/data/day_time_wildfire_v2_2192/usedImages.txt"

trainImagesPath = '/home/enrique/tfm/data/SF_dataset_resized_12620/images'
trainGTPath = '/home/enrique/tfm/data/SF_dataset_resized_12620/annotations'
trainUsePath = "/home/enrique/tfm/data/SF_dataset_resized_12620/usedImages.txt"

trainData = reader.readAndLoadData( trainImagesPath, trainGTPath, trainUsePath )
trainImages = [ img.data for img in trainData ]
trainBbox = [ [img.pointA.x, img.pointA.y, img.pointB.x, img.pointB.y] for img in trainData ]

testData = reader.readAndLoadData( testImagesPath, testGTPath, testUsePath )
testImages = [ img.data for img in testData ]
testBbox = [ [img.pointA.x, img.pointA.y, img.pointB.x, img.pointB.y] for img in testData ]

trainBbox = np.array( trainBbox, dtype = "float32")
testBbox = np.array( testBbox, dtype = "float32")

trainImages = np.array( trainImages, dtype = "float32") / 255.0
testImages = np.array( testImages, dtype = "float32") / 255.0 

print( "Hola" )

vgg = VGG16( weights="imagenet", include_top=False, input_tensor=Input( shape=(640, 480, 3) ) )
print( vgg.summary() )
vgg.trainable = False

flatten = vgg.output
flatten = Flatten()(flatten)

bboxHead = Dense(128, activation="relu" )(flatten)
bboxHead = Dense(64, activation="relu" )(bboxHead) 
bboxHead = Dense(32, activation="relu" )(bboxHead)
bboxHead = Dense(4, activation="sigmoid" )(bboxHead)

model = Model( inputs=vgg.input, outputs=bboxHead )

#initialise optimizer, compile the model, and show the model
opt = Adam( lr = 1e-4 )
model.compile( loss="mse", optimizer=opt)
print( model.summary() )

H = model.fit( trainImages, trainBbox, 
    batch_size=5,
    epochs=5,
    verbose=1)

print( "[INFO] Saving trained model to disk" )
model.save( outPath + "model", save_format="h5" )

