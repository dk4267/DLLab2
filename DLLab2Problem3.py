from PIL import Image
from keras.utils import to_categorical
from matplotlib import pyplot
import glob
from numpy import array
import numpy as np
from skimage.transform import resize
import os
import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.optimizers import SGD

train_images = []
train_labels = []
test_images = []
test_labels = []


def getImagesFromDir(path, label):
    number_im = len(os.listdir(path))
    i = 0

    path = path + '*.jpg'
    for filename in glob.glob(path):
        im = array(Image.open(filename).convert("L"))
        resized_im = resize(im, (100, 100))
        if i < int(number_im * 0.8):
            train_images.append(resized_im)
            train_labels.append(label)
        else:
            test_images.append(resized_im)
            test_labels.append(label)
        i += 1


# numerical labels dict
label_dict = {0: 'airplane', 1: 'car', 2: 'cat', 3: 'dog', 4: 'flower', 5: 'fruit', 6: 'motorbike', 7: 'person'}
# Load images in as int arrays
getImagesFromDir('natural_images/airplane/', 0)
getImagesFromDir('natural_images/car/', 1)
getImagesFromDir('natural_images/cat/', 2)
getImagesFromDir('natural_images/dog/', 3)
getImagesFromDir('natural_images/flower/', 4)
getImagesFromDir('natural_images/fruit/', 5)
getImagesFromDir('natural_images/motorbike/', 6)
getImagesFromDir('natural_images/person/', 7)
train_images = array(train_images)
train_labels = array(train_labels)
test_images = array(test_images)
test_labels = array(test_labels)

# reshape dataset to have a single channel
trainX = train_images.reshape((train_images.shape[0], 100, 100, 1))
testX = test_images.reshape((test_images.shape[0], 100, 100, 1))

# one hot encode target values
trainY = to_categorical(train_labels)
testY = to_categorical(test_labels)

# convert from integers to floats
train_norm = trainX.astype('float32')
test_norm = testX.astype('float32')
# normalize to range 0-1
trainX = train_norm / 255.0
testX = test_norm / 255.0

# define model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(100,100,1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dense(8, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# fit model
history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=2)
# evaluate model
_, acc = model.evaluate(testX, testY, verbose=0)
print('> %.3f' % acc)

# plot loss
pyplot.subplot(2, 1, 1)
pyplot.title('Cross Entropy Loss')
pyplot.plot(history.history['loss'], color='blue', label='train')
pyplot.plot(history.history['val_loss'], color='orange', label='test')
# plot accuracy
pyplot.subplot(2, 1, 2)
pyplot.title('Classification Accuracy')
pyplot.plot(history.history['accuracy'], color='blue', label='train')
pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
pyplot.show()