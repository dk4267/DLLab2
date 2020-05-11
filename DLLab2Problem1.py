import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
import matplotlib.pyplot as plt
from keras import optimizers

#read in data from csv, designate x and y data
dataset = pd.read_csv("column_2C.csv", header=0, encoding='ascii').dropna()

class_map = {'Normal': 0, 'Abnormal': 1}
dataset['class'] = dataset['class'].map(class_map)

ydata = dataset['class']
xdata = dataset.drop(['class'], axis=1)

#standard scaler for x data
sc = StandardScaler()
sc.fit(xdata)
X_scaled_array = sc.transform(xdata)
#train test split
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled_array, ydata,
                                                    test_size=0.25, random_state=87)
adam = optimizers.adam(lr=0.0001)
#build model
np.random.seed(155)
model = Sequential()
model.add(Dense(15, input_dim=6, activation='relu'))
model.add(Dense(1, activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['acc'])

#fit model and print accuracy and loss results
history = model.fit(X_train, Y_train, batch_size=64, epochs=100, initial_epoch=0)
print(model.summary())
print(model.evaluate(X_test, Y_test))

#plot history for accuracy and loss
plt.plot(history.history['acc'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#initial values: learning rate = 0.001, batch size = 32, optimizer = adam,
#activation = relu for first, sigmoid for second
#loss = 0.288, accuracy = 0.872

#change learning rate to 0.0001
#loss = 0.582, accuracy = 0.654
#not enough time allotted to learn

#change learning rate to 0.01
#loss = 0.297, accuracy = 0.872
#very slight increase in loss, no real difference

#initial learning rate, change batch size to 64
#loss = 0.320, accuracy = 0.894
#higher loss but higher accuracy

#change batch size to 16
#loss = 0.281, accuracy = 0.884
#results are slightly better than original

#initial batch size, change optimizer to SGD
#loss = 0.358, accuracy = 0.885
#higher loss, slightly higher accuracy

#initial optimizer, change activation for second layer to softmax
#loss = 3.519, accuracy = 0.769
#way more loss - not made for binary target data

