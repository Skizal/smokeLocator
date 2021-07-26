from tensorflow import keras

from keras.applications import VGG16
from keras.models import Model
from keras.layers import Dense, Flatten, Input
from keras.optimizers import Adam
import numpy as np

import dataReader as reader
from utils import ImageData, Point, Configuration


trainData = reader.readAndLoadData( Configuration.trainImages, Configuration.trainGT, Configuration.trainUsage )
trainImages = [ img.data for img in trainData ]
trainBbox = [ [ img.box.min.x, img.box.min.y, img.box.max.x, img.box.max.y ] for img in trainData ]

testData = reader.readAndLoadData( Configuration.testImages, Configuration.testGT, Configuration.testUsage )
testImages = [ img.data for img in testData ]
testBbox = [ [ img.box.min.x, img.box.min.y, img.box.max.x, img.box.max.y ] for img in testData ]

trainBbox = np.array( trainBbox, dtype = "float32")
testBbox = np.array( testBbox, dtype = "float32")

trainImages = np.array( trainImages, dtype = "float32") / 255.0
testImages = np.array( testImages, dtype = "float32") / 255.0 

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

print( "[INFO] Saving trained model to disk: " + Configuration.output + "model" )
model.save( Configuration.output + "model", save_format="h5" )

