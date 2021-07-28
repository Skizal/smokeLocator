from tensorflow import keras
import tensorflow_addons as tfa

from keras.applications import VGG16
from keras.models import Model
from keras.layers import Dense, Flatten, Input
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np


import dataReader as reader
from utils import *

'''
testData = reader.readAndLoadData( Configuration.testImages, Configuration.testGT, Configuration.testUsage )
testImages = [ img.data for img in testData ]
testBbox = [ [ img.box.min.x, img.box.min.y, img.box.max.x, img.box.max.y ] for img in testData ]

testBbox = np.array( testBbox, dtype = "float32")
testImages = np.array( testImages, dtype = "float32") / 255.0 
'''

trainData = reader.readAndLoadData( Configuration.trainImages, Configuration.trainGT, Configuration.trainUsage )
trainImages = [ img.data for img in trainData ]
trainBbox = [ [ img.box.min.x, img.box.min.y, img.box.max.x, img.box.max.y ] for img in trainData ]

trainBbox = np.array( trainBbox, dtype = "float32")
trainImages = np.array( trainImages, dtype = "float32") / 255.0

valImages = trainImages[400:]
valBbox = trainBbox[400:]

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
opt = Adam( lr = Configuration.learningRate )
model.compile( optimizer = opt, loss = [ciouCoef] )
print( model.summary() )

H = model.fit( trainImages, trainBbox,
    validation_data= ( valImages, valBbox ),
    batch_size = Configuration.batchSize,
    epochs = Configuration.nEpochs,
    verbose=1)

# plot the model training history
N = Configuration.nEpochs
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history['loss'], label="loss")
plt.title("Bounding Box Regression Loss on Training Set")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")

print( "[INFO] Saving trained model to disk: " + Configuration.modelPath + "model" )
model.save( Configuration.modelPath + "model", save_format="h5" )

