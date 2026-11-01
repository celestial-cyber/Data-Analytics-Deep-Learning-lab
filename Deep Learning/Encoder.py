# An Autoencoder is an unsupervised deep learning algorithm
# used to compress data and reconstruct the original input.
# It consists of two main parts:
# Encoder: Compresses input data into a smaller representation.
# Decoder: Reconstructs the original data from the compressed representation.
# This program uses a Variational Autoencoder (VAE) with CNN layers
# to learn and generate handwritten digit images from the MNIST dataset.
# The latent dimension is 2, which represents compressed data.

# Step 1: Start the program.
# Step 2: Import the required libraries and MNIST dataset.
# Step 3: Normalize the image data.
# Step 4: Define the Encoder to compress input images.
# Step 5: Define the Decoder to reconstruct images.
# Step 6: Apply the sampling technique to generate latent vectors.
# Step 7: Compile and train the Variational Autoencoder.
# Step 8: Generate new images from random latent vectors.
# Step 9: Display the generated images.
# Step 10: Stop the program.

# Output:
# Epoch 1/30
# Loss: 216.42

# Epoch 2/30
# Loss: 187.62

# ...

# Epoch 30/30
# Loss: 147.65

# A 10x10 grid of generated handwritten digit images is displayed.

# Import libraries
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model

# Load and normalize MNIST dataset
(x_train, _), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train[..., None].astype("float32") / 255

# Define Encoder
inp = layers.Input(shape=(28, 28, 1))
x = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(inp)
x = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
x = layers.Flatten()(x)
x = layers.Dense(16, activation="relu")(x)

mean = layers.Dense(2)(x)
log_var = layers.Dense(2)(x)
encoder = Model(inp, [mean, log_var])

# Define Decoder
z = layers.Input(shape=(2,))
x = layers.Dense(7*7*64, activation="relu")(z)
x = layers.Reshape((7, 7, 64))(x)
x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)
x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)
out = layers.Conv2DTranspose(1, 3, padding="same", activation="sigmoid")(x)
decoder = Model(z, out)

# Define VAE
class VAE(Model):
    def __init__(self):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]

        with tf.GradientTape() as tape:
            m, v = self.encoder(data)
            z = m + tf.exp(0.5 * v) * tf.random.normal(tf.shape(m))
            recon = self.decoder(z)

            rec_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(data, recon)
            ) * 784

            kl_loss = -0.5 * tf.reduce_mean(
                1 + v - tf.square(m) - tf.exp(v)
            )

            loss = rec_loss + kl_loss

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return {"loss": loss, "reconstruction_loss": rec_loss,
                "kl_loss": kl_loss}

# Compile and train
vae = VAE()
vae.compile(optimizer="adam")
vae.fit(x_train, epochs=30, batch_size=128)

# Generate new images
z = np.random.normal(size=(100, 2))
images = decoder.predict(z, verbose=0)

# Display images in a 10x10 grid
plt.figure(figsize=(10, 10))

for i, img in enumerate(images):
    plt.subplot(10, 10, i+1)
    plt.imshow(img.squeeze(), cmap="Greys_r")
    plt.axis("off")

plt.show()