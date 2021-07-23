import tensorflow
from tensorflow import keras

from keras.applications import VGG16
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam
import numpy as np

import dataReader as reader
from utils import ImageData, Point

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

#trainImages = np.array( trainImages, dtype = "float32") / 255.0
testImages = np.array( testImages, dtype = "float32") / 255.0 


print( "Hola" )


'''
train_images = train_images.reshape( (60000, 28, 28, 1) )
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape( (10000, 28, 28, 1) )
test_images = test_images.astype( 'float32' )

train_labels = to_categorical( train_labels )
test_labels = to_categorical( test_labels )


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1) ) )
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

print(model.summary())

model.compile( optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'] )

model.fit( train_images, train_labels, epochs=5, batch_size=64 ) 

score = model.evaluate( test_images, test_labels )
print('Test loss:', score[0])
print('Test accuracy:', score[1])
'''