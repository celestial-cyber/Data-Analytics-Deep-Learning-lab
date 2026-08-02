# =============================================================================
#                        EXPERIMENT NO: 8
# =============================================================================
#
# TITLE:
#
# Applying Generative Adversarial Networks (GANs)
# for Image Generation and Unsupervised Learning
#
# =============================================================================
#
# AIM:
#
# To implement a Generative Adversarial Network (GAN) using TensorFlow/Keras
# for generating handwritten digit images and understanding its role in
# unsupervised learning.
#
# =============================================================================
# THEORY
# =============================================================================
#
# What is GAN?
# -----------------------------------------------------------------------------
#
# GAN stands for Generative Adversarial Network.
#
# It is a Deep Learning model proposed by Ian Goodfellow in 2014.
#
# GAN consists of TWO Neural Networks:
#
#          Random Noise
#               │
#               ▼
#          Generator
#               │
#               ▼
#        Fake Generated Image
#               │
#               ▼
#         Discriminator
#          ▲          │
#          │          ▼
#     Real Images   Fake Images
#
# Generator and Discriminator compete against each other.
#
# Hence the name:
#
# Generative
# +
# Adversarial
#
# =============================================================================
#
# What is Generator?
#
# Generator creates fake images from random noise.
#
# Initially:
#
# Generated images are meaningless.
#
# After training:
#
# Generator learns how real images look and creates realistic images.
#
# =============================================================================
#
# What is Discriminator?
#
# Discriminator is a Binary Classifier.
#
# It predicts:
#
# Real Image → 1
#
# Fake Image → 0
#
# =============================================================================
#
# Goal of GAN
#
# Generator tries to fool the Discriminator.
#
# Discriminator tries to detect fake images.
#
# Eventually,
#
# Generator becomes so good that
# fake images look almost real.
#
# =============================================================================
#
# Applications of GAN
#
# • Face Generation
# • Image Generation
# • Anime Generation
# • Medical Imaging
# • Image Super Resolution
# • Image Colorization
# • Deepfake Generation
# • Data Augmentation
#
# =============================================================================
#
# Why is GAN Unsupervised?
#
# GAN does not require labels.
#
# It only learns the distribution of data.
#
# Therefore,
# GAN belongs to Unsupervised Learning.
#
# =============================================================================
#
# IMPORT LIBRARIES
# =============================================================================

import numpy as np

# NumPy is used for numerical operations.

import tensorflow as tf

# TensorFlow is used for Deep Learning.

from tensorflow.keras.models import Sequential

# Sequential model builds the network layer by layer.

from tensorflow.keras.layers import Dense, Flatten, Reshape

# Dense: Fully connected layer.
# Flatten: Converts image to 1D.
# Reshape: Converts vector back into image.

from tensorflow.keras.datasets import mnist

# MNIST dataset containing handwritten digits.

# =============================================================================
# LOAD DATASET
# =============================================================================

# Load MNIST dataset.

(X_train, _), (_, _) = mnist.load_data()

# Labels are ignored because GAN training does not require them.

print("Dataset Shape :", X_train.shape)

# =============================================================================
# NORMALIZATION
# =============================================================================
#
# Pixel values:
#
# 0 → 255
#
# Convert into:
#
# -1 → +1
#
# Why?
#
# Generator uses tanh activation.
#
# tanh outputs values between
#
# -1 and +1
#
# Hence,
# training becomes stable.
#
# =============================================================================

X_train = (X_train.astype('float32') - 127.5) / 127.5

# =============================================================================
# FLATTEN IMAGES
# =============================================================================
#
# Original Shape:
#
# (60000,28,28)
#
# New Shape:
#
# (60000,784)
#
# =============================================================================

X_train = X_train.reshape(60000,784)

# =============================================================================
# CREATE GENERATOR
# =============================================================================
#
# Generator converts
#
# Random Noise
#
# into
#
# Fake Images
#
# =============================================================================

generator = Sequential()

# First Dense Layer
#
# Input:
#
# Random Noise Vector (100 values)
#
generator.add(Dense(256, activation='relu', input_dim=100))

# Second Dense Layer

generator.add(Dense(512, activation='relu'))

# Third Dense Layer

generator.add(Dense(1024, activation='relu'))

# Output Layer
#
# 784 neurons because
#
# 28 × 28 = 784 pixels
#
# tanh activation
#
generator.add(Dense(784, activation='tanh'))

print("\nGenerator Summary")

generator.summary()

# =============================================================================
# CREATE DISCRIMINATOR
# =============================================================================
#
# Discriminator receives
#
# Real Image
#
# OR
#
# Fake Image
#
# and predicts
#
# Real or Fake
#
# =============================================================================

discriminator = Sequential()

discriminator.add(Dense(512,
                         activation='relu',
                         input_dim=784))

discriminator.add(Dense(256,
                         activation='relu'))

# Output Layer
#
# One neuron
#
# Binary Classification
#
# sigmoid
#
discriminator.add(Dense(1,
                         activation='sigmoid'))

# Compile Discriminator

discriminator.compile(

    optimizer='adam',

    loss='binary_crossentropy',

    metrics=['accuracy']

)

print("\nDiscriminator Summary")

discriminator.summary()

# =============================================================================
# CREATE GAN MODEL
# =============================================================================
#
# During Generator training,
#
# Discriminator weights are frozen.
#
# Only Generator learns.
#
# =============================================================================

discriminator.trainable = False

gan = Sequential()

gan.add(generator)

gan.add(discriminator)

gan.compile(

    optimizer='adam',

    loss='binary_crossentropy'

)

print("\nGAN Summary")

gan.summary()

# =============================================================================
# END OF PART 1
# =============================================================================
#
# =============================================================================
# STEP 1 : DEFINE TRAINING PARAMETERS
# =============================================================================
#
# Noise Dimension:
# Random vector size given as input to Generator.
#
# Batch Size:
# Number of images processed in one training iteration.
#
# Epochs:
# Number of complete training cycles.
#
# =============================================================================

noise_dim = 100

batch_size = 64

epochs = 1000

# =============================================================================
# STEP 2 : TRAIN GAN
# =============================================================================
#
# GAN training consists of TWO stages:
#
# Stage 1:
# Train Discriminator
#
# Stage 2:
# Train Generator
#
# These two networks compete against each other.
#
# =============================================================================

for epoch in range(epochs):

    # -------------------------------------------------------------------------
    # TRAIN DISCRIMINATOR
    # -------------------------------------------------------------------------

    # Select random real images.
    idx = np.random.randint(0, X_train.shape[0], batch_size)

    real_images = X_train[idx]

    # Generate random noise.
    noise = np.random.normal(0, 1, (batch_size, noise_dim))

    # Generator creates fake images.
    fake_images = generator.predict(noise, verbose=0)

    # Labels for real images = 1
    real_labels = np.ones((batch_size, 1))

    # Labels for fake images = 0
    fake_labels = np.zeros((batch_size, 1))

    # Train discriminator on real images.
    d_loss_real = discriminator.train_on_batch(real_images, real_labels)

    # Train discriminator on fake images.
    d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)

    # Average discriminator loss.
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # -------------------------------------------------------------------------
    # TRAIN GENERATOR
    # -------------------------------------------------------------------------

    # Generate another batch of random noise.
    noise = np.random.normal(0, 1, (batch_size, noise_dim))

    # Generator wants discriminator to classify
    # fake images as REAL.
    valid_labels = np.ones((batch_size, 1))

    # Train Generator through GAN model.
    g_loss = gan.train_on_batch(noise, valid_labels)

    # -------------------------------------------------------------------------
    # PRINT TRAINING PROGRESS
    # -------------------------------------------------------------------------

    if epoch % 100 == 0:

        print("\nEpoch :", epoch)

        print("Discriminator Loss :", d_loss[0])

        print("Discriminator Accuracy :", d_loss[1] * 100)

        print("Generator Loss :", g_loss)

# =============================================================================
# STEP 3 : GENERATE NEW IMAGES
# =============================================================================
#
# After training,
# Generator can create new handwritten digits.
#
# =============================================================================

noise = np.random.normal(0, 1, (5, noise_dim))

generated_images = generator.predict(noise)

print("\nGenerated Image Shape :")

print(generated_images.shape)

# =============================================================================
# STEP 4 : RESHAPE GENERATED IMAGES
# =============================================================================
#
# Current Shape:
#
# (5,784)
#
# Convert into:
#
# (5,28,28)
#
# =============================================================================

generated_images = generated_images.reshape(5, 28, 28)

print("\nReshaped Image Shape :")

print(generated_images.shape)

# =============================================================================
# EXPECTED OUTPUT
# =============================================================================
#
# Dataset Shape:
#
# (60000,28,28)
#
# Epoch 0
#
# Discriminator Loss : 0.67
#
# Generator Loss : 1.42
#
# ------------------------------
#
# Epoch 100
#
# Discriminator Accuracy : 82%
#
# ------------------------------
#
# Epoch 500
#
# Discriminator Accuracy : 91%
#
# ------------------------------
#
# Epoch 1000
#
# Generator produces realistic
# handwritten digits.
#
# Generated Image Shape:
#
# (5,784)
#
# Reshaped Image Shape:
#
# (5,28,28)
#
# =============================================================================
# APPLICATIONS OF GAN
# =============================================================================
#
# 1. Face Generation
#
# 2. AI Art Generation
#
# 3. Deepfake Videos
#
# 4. Medical Image Generation
#
# 5. Image Colorization
#
# 6. Super Resolution
#
# 7. Image Inpainting
#
# 8. Data Augmentation
#
# 9. Style Transfer
#
# 10. Synthetic Data Generation
#
# =============================================================================
# ADVANTAGES
# =============================================================================
#
# • Generates realistic images.
#
# • Learns without labeled data.
#
# • Useful for data augmentation.
#
# • Produces high-quality synthetic data.
#
# • Helps when real data is limited.
#
# =============================================================================
# LIMITATIONS
# =============================================================================
#
# • Difficult to train.
#
# • Training is unstable.
#
# • Can suffer from Mode Collapse
#   (generator produces very similar outputs).
#
# • Requires large computational resources.
#
# =============================================================================
# COMMON ERRORS
# =============================================================================
#
# Error:
# Generator produces only noise.
#
# Solution:
# Train for more epochs or tune learning rate.
#
# ------------------------------------------------------------
#
# Error:
# Discriminator becomes too strong.
#
# Solution:
# Balance training between generator and discriminator.
#
# ------------------------------------------------------------
#
# Error:
# Mode Collapse.
#
# Solution:
# Use improved GAN variants (e.g., WGAN, DCGAN) or adjust training.
#
# =============================================================================
# VIVA QUESTIONS WITH ANSWERS
# =============================================================================
#
# Q1. What is GAN?
# Ans:
# GAN (Generative Adversarial Network) is a deep learning model consisting of
# a Generator and a Discriminator that compete to generate realistic data.
#
# ------------------------------------------------------------
#
# Q2. Who proposed GAN?
# Ans:
# Ian Goodfellow in 2014.
#
# ------------------------------------------------------------
#
# Q3. What are the two main components of GAN?
# Ans:
# Generator and Discriminator.
#
# ------------------------------------------------------------
#
# Q4. What is the role of the Generator?
# Ans:
# It creates fake data from random noise.
#
# ------------------------------------------------------------
#
# Q5. What is the role of the Discriminator?
# Ans:
# It classifies input data as real or fake.
#
# ------------------------------------------------------------
#
# Q6. Why is GAN called 'Adversarial'?
# Ans:
# Because the Generator and Discriminator compete against each other during
# training.
#
# ------------------------------------------------------------
#
# Q7. Why is GAN considered unsupervised?
# Ans:
# It learns the data distribution without requiring class labels.
#
# ------------------------------------------------------------
#
# Q8. What is the input to the Generator?
# Ans:
# A random noise vector (latent vector).
#
# ------------------------------------------------------------
#
# Q9. Why use tanh activation in the Generator output?
# Ans:
# Because the input images are normalized to the range [-1, 1].
#
# ------------------------------------------------------------
#
# Q10. Why use sigmoid in the Discriminator output?
# Ans:
# It outputs a probability between 0 and 1 for binary classification.
#
# ------------------------------------------------------------
#
# Q11. What loss function is used in GAN?
# Ans:
# Binary Crossentropy.
#
# ------------------------------------------------------------
#
# Q12. What optimizer is commonly used?
# Ans:
# Adam optimizer.
#
# ------------------------------------------------------------
#
# Q13. What is Mode Collapse?
# Ans:
# A problem where the Generator produces very similar outputs repeatedly.
#
# ------------------------------------------------------------
#
# Q14. Mention applications of GAN.
# Ans:
# Face generation, image synthesis, super-resolution, style transfer,
# data augmentation, and medical imaging.
#
# ------------------------------------------------------------
#
# Q15. What is a latent vector?
# Ans:
# A random vector that serves as the input to the Generator.
#
# ------------------------------------------------------------
#
# Q16. Why is the Discriminator frozen while training the Generator?
# Ans:
# To update only the Generator's weights while keeping the Discriminator fixed.
#
# ------------------------------------------------------------
#
# Q17. What is DeepFake technology?
# Ans:
# A GAN-based application that generates realistic synthetic faces or videos.
#
# ------------------------------------------------------------
#
# Q18. What dataset is used in this experiment?
# Ans:
# MNIST handwritten digit dataset.
#
# ------------------------------------------------------------
#
# Q19. Why normalize images before training?
# Ans:
# To stabilize training and match the output range of the tanh activation.
#
# ------------------------------------------------------------
#
# Q20. Name some improved GAN variants.
# Ans:
# DCGAN, Conditional GAN (CGAN), CycleGAN, StyleGAN, and WGAN.
#


#Q: Why do we freeze (discriminator.trainable = False) the Discriminator while training the Generator?

#Answer:
#When training the Generator, we want to update only the Generator's weights. The Discriminator is kept 
#fixed so it acts as a stable evaluator, providing feedback on how realistic the generated images are. 
##If both networks were updated simultaneously during this step, the Generator would not receive a consistent 
#learning signal
# =============================================================================
# END OF EXPERIMENT
# =============================================================================

