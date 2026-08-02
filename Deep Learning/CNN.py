# =====================================================
# AIM:
# To implement a Convolutional Neural Network (CNN)
# for solving a computer vision problem such as image
# classification.
# =====================================================

# THEORY:
# CNN (Convolutional Neural Network) is a deep learning
# algorithm specially designed for processing images.
#
# Why CNN?
# - Automatically extracts features from images.
# - Requires less manual feature engineering.
# - More accurate than traditional ANN for images.
#
# Applications:
# - Face Recognition
# - Medical Image Analysis
# - Self-driving Cars
# - Object Detection
# - OCR
# =====================================================

# Import TensorFlow library
# Why?
# TensorFlow provides tools to build and train deep learning models.
import tensorflow as tf

# Import Keras Sequential model
# Why?
# Sequential helps us create neural networks layer by layer.
from tensorflow.keras.models import Sequential

# Import CNN layers
# Conv2D -> Performs convolution on images.
# MaxPooling2D -> Reduces image size while retaining important features.
# Flatten -> Converts 2D feature maps into a 1D vector.
# Dense -> Fully connected layer used for classification.
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# =====================================================
# Load Dataset
# Why?
# CIFAR-10 contains 60,000 color images across 10 classes.
# It is commonly used to train CNN models.
# =====================================================
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize pixel values.
# Why?
# Pixel values range from 0 to 255.
# Dividing by 255 scales them between 0 and 1.
# This improves training speed and stability.
x_train = x_train / 255.0
x_test = x_test / 255.0

# =====================================================
# Build CNN Model
# =====================================================

model = Sequential()

# First Convolution Layer
# 32 filters detect simple features like edges.
# Kernel size (3x3) scans small regions.
# ReLU introduces non-linearity.
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)))

# Max Pooling
# Reduces image dimensions.
# Keeps the most important features.
model.add(MaxPooling2D(pool_size=(2,2)))

# Second Convolution Layer
# 64 filters detect more complex patterns.
model.add(Conv2D(64, (3,3), activation='relu'))

# Second Pooling Layer
model.add(MaxPooling2D(pool_size=(2,2)))

# Flatten Layer
# Converts feature maps into a single vector
# before feeding into Dense layers.
model.add(Flatten())

# Hidden Dense Layer
# Learns high-level patterns from extracted features.
model.add(Dense(128, activation='relu'))

# Output Layer
# 10 neurons because CIFAR-10 has 10 classes.
# Softmax converts outputs into probabilities.
model.add(Dense(10, activation='softmax'))

# =====================================================
# Compile Model
# =====================================================

# Adam optimizer updates weights efficiently.
# Sparse Categorical Crossentropy is suitable
# because labels are integers.
# Accuracy measures prediction performance.
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# =====================================================
# Train Model
# =====================================================

# epochs=10 means the entire dataset is used
# 10 times for learning.
model.fit(x_train, y_train, epochs=10)

# =====================================================
# Evaluate Model
# =====================================================

# Tests the trained model on unseen images.
test_loss, test_acc = model.evaluate(x_test, y_test)

# Print Accuracy
print("Test Accuracy:", test_acc)

# =====================================================
# VIVA QUESTIONS
# =====================================================
#
# Q1. What is CNN?
# Ans: CNN is a deep learning model specially designed
# for image processing.
#
# Q2. Why use ReLU?
# Ans: It removes negative values and helps the model
# learn complex patterns faster.
#
# Q3. Why MaxPooling?
# Ans: To reduce computation while preserving important
# image features.
#
# Q4. Why Flatten?
# Ans: Dense layers require 1D input.
#
# Q5. Why Softmax?
# Ans: It converts outputs into probabilities whose sum is 1.
#
# Q6. Difference between ANN and CNN?
# Ans:
# ANN uses all neurons connected.
# CNN uses convolution filters for feature extraction.
#
# Time Complexity:
# O(Number of Filters × Image Size × Kernel Size)
#
# Applications:
# Face Recognition
# Medical Imaging
# Autonomous Vehicles
# OCR
# Satellite Image Analysis
#
# =====================================================