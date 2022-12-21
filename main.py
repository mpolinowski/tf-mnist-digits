import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random

# download mnist dataset
from tensorflow.keras.datasets import mnist
(X_train, y_train),(X_test, y_test) = mnist.load_data()

## normalize the data
X_train = X_train / 255
X_test = X_test / 255

# inspect the dataset
# print(X_train.shape)
## (60000, 28, 28) => 60k monochrome 28x28 pixel images
# print(X_test.shape)
## (10000, 28, 28) => 10k monochrome 28x28 pixel images


# # select random training image
# i = random.randint(1,60000)
# # inspect single image from dataset
# # reshape and plot the image
# plt.imshow( X_train[i] , cmap = 'gray')
# label = y_train[i]
# plt.show()


# # inspect a larger set of images
# ## create a 15x15 grid
# W_grid = 15
# L_grid = 15
# fig, axes = plt.subplots(L_grid, W_grid)
# fig, axes = plt.subplots(L_grid, W_grid, figsize = (17,17))
# ## flaten the 15 x 15 matrix into 225 array
# axes = axes.ravel()
# ## get the length of the training dataset
# n_training = len(X_train)

# for i in np.arange(0, W_grid * L_grid):
#     # select a random number
#     index = np.random.randint(0, n_training)
#     # read and display an image with the selected index    
#     axes[i].imshow( X_train[index] )
#     axes[i].set_title(y_train[index], fontsize = 8)
#     axes[i].axis('off')

# plt.subplots_adjust(hspace=0.4)
# plt.tight_layout()
# plt.show()


# adding noise

# ## creating noise
# ### create a 28x28 matrix of random number
# noise_factor = 0.3
# added_noise = noise_factor * np.random.randn(*(28,28))
# ## test noise map
# plt.imshow(added_noise)
# plt.show()


# ## adding noise to an image
# ### how much noise do we need
# noise_factor = 0.2
# ### select a random image
# sample_image = X_train[random.randint(1,60000)]
# ### and add noise map to it
# noisy_sample_image = sample_image + noise_factor * np.random.randn(*(28,28))
# ### by adding the noise we lost our normalization =>
# ### clip values outside of the range 0-1
# noisy_sample_image = np.clip(noisy_sample_image, 0., 1.)

# # previs noisy image
# plt.imshow(noisy_sample_image, cmap="gray")
# plt.show()


# add noise to all images in training dataset
X_train_noisy = []
noise_factor = 0.2

for sample_image in X_train:
  sample_image_noisy = sample_image + noise_factor * np.random.randn(*(28,28))
  sample_image_noisy = np.clip(sample_image_noisy, 0., 1.)
  X_train_noisy.append(sample_image_noisy)

# Convert from list to np array
X_train_noisy = np.array(X_train_noisy)


# add noise to all images in testing dataset
X_test_noisy = []
noise_factor = 0.4

for sample_image in X_test:
  sample_image_noisy = sample_image + noise_factor * np.random.randn(*(28,28))
  sample_image_noisy = np.clip(sample_image_noisy, 0., 1.)
  X_test_noisy.append(sample_image_noisy)

# Convert from list to array
X_test_noisy = np.array(X_test_noisy)


# # show random images from noisy datasets
# plt.imshow(X_train_noisy[random.randint(1,60000)], cmap="gray")
# plt.show()
# plt.imshow(X_test_noisy[random.randint(1,60000)], cmap="gray")
# plt.show()


# build autoencoder model
autoencoder = tf.keras.models.Sequential()

# build the encoder CNN
autoencoder.add(tf.keras.layers.Conv2D(16, (3,3), strides=1, padding="same", input_shape=(28, 28, 1)))
autoencoder.add(tf.keras.layers.MaxPooling2D((2,2), padding="same"))

autoencoder.add(tf.keras.layers.Conv2D(8, (3,3), strides=1, padding="same"))
autoencoder.add(tf.keras.layers.MaxPooling2D((2,2), padding="same"))

# representation layer
autoencoder.add(tf.keras.layers.Conv2D(8, (3,3), strides=1, padding="same"))

# build the decoder CNN 
autoencoder.add(tf.keras.layers.UpSampling2D((2, 2)))
autoencoder.add(tf.keras.layers.Conv2DTranspose(8,(3,3), strides=1, padding="same"))

autoencoder.add(tf.keras.layers.UpSampling2D((2, 2)))
autoencoder.add(tf.keras.layers.Conv2DTranspose(1, (3,3), strides=1, activation='sigmoid', padding="same"))

# compile model
autoencoder.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.001))
autoencoder.summary()

# fit model to dataset
autoencoder.fit(X_train_noisy.reshape(-1, 28, 28, 1),          
          X_train.reshape(-1, 28, 28, 1), 
          epochs=10, 
          batch_size=200)

# test training
# take 15 images from noisy test set and predict de-noised state
denoised_images = autoencoder.predict(X_test_noisy[:15].reshape(-1, 28, 28, 1))
# plot noisy input vs denoised output
fig, axes = plt.subplots(nrows=2, ncols=15, figsize=(30,6))
for images, row in zip([X_test_noisy[:15], denoised_images], axes):
    for img, ax in zip(images, row):
        ax.imshow(img.reshape((28, 28)), cmap='gray')

plt.show()