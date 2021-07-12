import tensorflow
from tensorflow import keras

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras import backend as K


# the data, split between train and test sets
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

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