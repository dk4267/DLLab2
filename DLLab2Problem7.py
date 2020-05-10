from keras.layers import Input, Dense
from keras.datasets import mnist
from keras.models import Model
import matplotlib.pyplot as plt

# Model configuration
img_width, img_height = 28, 28
initial_dimension = img_width * img_height
# Load MNIST dataset
(input_train, target_train), (input_test, target_test) = mnist.load_data()
# Reshape data
input_train = input_train.reshape(input_train.shape[0], initial_dimension)
input_test = input_test.reshape(input_test.shape[0], initial_dimension)
input_shape = (initial_dimension, )
# Parse numbers as floats
input_train = input_train.astype('float32')
input_test = input_test.astype('float32')
# Normalize data
input_train = input_train / 255
input_test = input_test / 255
# Define the layers
encoded_dim = 50
inputs = Input(shape=input_shape)
encoding_layer = Dense(encoded_dim, activation='relu', kernel_initializer='he_normal')(inputs)
decoding_layer = Dense(initial_dimension, activation='sigmoid')(encoding_layer)
# Instantiate the autoencoder
autoencoder = Model(inputs, decoding_layer, name='full_autoencoder')
# Instantiate the encoder
encoder = Model(inputs, encoding_layer, name='encoder')
# Instantiate the decoder
encoded_input = Input(shape=(encoded_dim, ))
# Compile the autoencoder
encoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
# Fit data
autoencoder.fit(input_train, input_train, epochs=10, batch_size=128, validation_split=0.2)

# Visualize a sample
input_sample = input_test[:1]
reconstruction = autoencoder.predict([input_sample])
# Plot the sample input and reconstruction
fig, axes = plt.subplots(1, 2)
fig.set_size_inches(6, 3.5)
input_sample_reshaped = input_sample.reshape((img_width, img_height))
reconstruction_reshaped = reconstruction.reshape((img_width, img_height))
axes[0].imshow(input_sample_reshaped)
axes[0].set_title('Original image')
axes[1].imshow(reconstruction_reshaped)
axes[1].set_title('Reconstruction')
plt.show()
# Visualize encoded state
encoded_imgs = encoder.predict(input_test[:1])
plt.figure(figsize=(20, 8))
plt.imshow(encoded_imgs[0].reshape(2, 25).T)
plt.gray()
plt.show()
