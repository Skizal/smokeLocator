
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np


import dataReader as reader
from utils import *
from losses import *
import sys
import gc

for index, trainSet in enumerate( Configuration.trainImages ):
#     for lRate in Configuration.learningRate:
#         for batch in Configuration.batchSize:

    sys.stdout = open(str( Configuration.modelPath + "LOG_" + str(index) + "_" + str(Configuration.batchSize[1]) + "_" + str(Configuration.learningRate[0]) ), 'w')

    trainData = reader.readAndLoadData( Configuration.trainImages[index], Configuration.trainGT[index], Configuration.trainUsage[index], Configuration.limitImages )
    trainImages = [ img.data for img in trainData ]
    trainBbox = [ [ img.box.min.x, img.box.min.y, img.box.max.x, img.box.max.y ] for img in trainData ]

    trainBbox = np.array( trainBbox, dtype = "float32")
    trainImages = np.array( trainImages, dtype = "float32") / 255.0

    # valImages = trainImages[1650:]
    # valBbox = trainBbox[1650:]

    # trainImages = trainImages[:1650]
    # trainBbox = trainBbox[:1650]

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
    opt = Adam( lr = Configuration.learningRate[0] )
    model.compile( optimizer = opt, loss = [ciouCoef] )
    #print( model.summary() )

    H = model.fit( trainImages, trainBbox,
        batch_size = Configuration.batchSize[1],
        epochs = Configuration.nEpochs,
        verbose=2)

    for layer in model.layers[15:]:
        layer.trainable = True

    H = model.fit( trainImages, trainBbox,
        batch_size = Configuration.batchSize[1],
        epochs = Configuration.nEpochs,
        verbose=2)
    
        # H = model.fit( trainImages, trainBbox,
        # validation_data= ( valImages, valBbox ),
        # batch_size = Configuration.batchSize[1],
        # epochs = Configuration.nEpochs,
        # verbose=2)

    print( "[INFO] Saving trained model to disk: " + Configuration.modelPath + "model" )
    model.save( Configuration.modelPath + "model_" + str(index) + "_" + str(Configuration.batchSize[1]) + "_" + str( Configuration.learningRate[0] ), save_format="h5" )

    sys.stdout.close()

    del trainData
    del trainImages
    del trainBbox
    gc.collect()

