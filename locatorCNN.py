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
from losses import *


trainData = reader.readAndLoadData( Configuration.testImages, Configuration.testGT, Configuration.testUsage )
trainImages = [ img.data for img in trainData ]
trainBbox = [ [ img.box.min.x, img.box.min.y, img.box.max.x, img.box.max.y ] for img in trainData ]

trainBbox = np.array( trainBbox, dtype = "float32")
trainImages = np.array( trainImages, dtype = "float32") / 255.0

valImages = trainImages[1600:]
valBbox = trainBbox[1600:]

trainImages = trainImages[:1600]
trainBbox = trainBbox[:1600]

vgg = VGG16( weights="imagenet", include_top=False, input_tensor=Input( shape=( Configuration.xRes, Configuration.yRes, 3) ) )
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
model.compile( optimizer = opt, loss = [diouCoef] )
print( model.summary() )

H = model.fit( trainImages, trainBbox,
    validation_data= ( valImages, valBbox ),
    batch_size = Configuration.batchSize,
    epochs = Configuration.nEpochs,
    verbose=1)

for layer in model.layers[15:]:
	layer.trainable = True
# loop over the layers in the model and show which ones are trainable
# or not
for layer in model.layers:
	print("{}: {}".format(layer, layer.trainable))

H = model.fit( trainImages, trainBbox,
    validation_data= ( valImages, valBbox ),
    batch_size = Configuration.batchSize,
    epochs = Configuration.nEpochs,
    verbose=1)

# plot the model training history
'''
N = Configuration.nEpochs
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history['loss'], label="loss")
plt.title("Bounding Box Regression Loss on Training Set")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
'''

print( "[INFO] Saving trained model to disk: " + Configuration.modelPath + "model" )
model.save( Configuration.modelPath + "modelC", save_format="h5" )

