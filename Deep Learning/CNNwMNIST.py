# =============================================================================
# EXPERIMENT NO: 4
# TITLE: Image Classification on MNIST Dataset using Convolutional Neural Network
# =============================================================================

# AIM:
# To build and train a Convolutional Neural Network (CNN) for classifying
# handwritten digits (0–9) from the MNIST dataset using TensorFlow and Keras.

# =============================================================================
# THEORY:
#
# What is MNIST?
# ----------------
# MNIST (Modified National Institute of Standards and Technology) is one of the
# most popular datasets used for image classification.
#
# It contains:
# - 60,000 training images
# - 10,000 testing images
# - Each image is grayscale
# - Image size = 28 x 28 pixels
# - Classes = Digits 0 to 9
#
# Why CNN?
# --------
# CNN automatically extracts image features such as edges, curves,
# and patterns without manual feature engineering.
#
# CNN Architecture:
#
# Input Image
#      ↓
# Convolution Layer
#      ↓
# ReLU Activation
#      ↓
# Max Pooling
#      ↓
# Flatten Layer
#      ↓
# Fully Connected (Dense Layer)
#      ↓
# Output Layer (Softmax)
#
# Applications:
# - Digit Recognition
# - OCR
# - Face Recognition
# - Medical Imaging
# - Traffic Sign Recognition
# =============================================================================

# Import TensorFlow library.
# Why?
# TensorFlow provides deep learning functionalities.
import tensorflow as tf

# Import MNIST dataset.
from tensorflow.keras.datasets import mnist

# Import Sequential model.
# Sequential builds the neural network layer by layer.
from tensorflow.keras.models import Sequential

# Import CNN layers.
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# =============================================================================
# STEP 1: Load Dataset
# =============================================================================

# Load MNIST dataset.
# Returns:
# x_train -> Training images
# y_train -> Training labels
# x_test -> Testing images
# y_test -> Testing labels
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# =============================================================================
# STEP 2: Data Preprocessing
# =============================================================================

# Normalize pixel values.
# Original pixel values range from 0 to 255.
# Dividing by 255 converts them into values between 0 and 1.
# This improves training speed and accuracy.
x_train = x_train / 255.0
x_test = x_test / 255.0

# CNN expects 4D input:
# (Number of Images, Height, Width, Channels)
#
# Current shape:
# (60000,28,28)
#
# After reshape:
# (60000,28,28,1)
#
# 1 indicates grayscale image.
x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)

# =============================================================================
# STEP 3: Build CNN Model
# =============================================================================

# Create Sequential Model.
model = Sequential()

# -------------------------------------------------------------------------
# First Convolution Layer
#
# Conv2D Parameters:
#
# 32 -> Number of filters
# (3,3) -> Kernel size
# activation='relu'
#
# Purpose:
# Detect edges and simple features.
# -------------------------------------------------------------------------
model.add(Conv2D(32,(3,3),
                 activation='relu',
                 input_shape=(28,28,1)))

# -------------------------------------------------------------------------
# Max Pooling Layer
#
# Pool Size = 2 x 2
#
# Purpose:
# Reduces image dimensions.
# Keeps important features.
# Reduces computation.
# -------------------------------------------------------------------------
model.add(MaxPooling2D(pool_size=(2,2)))

# -------------------------------------------------------------------------
# Second Convolution Layer
#
# More filters means the network can learn more complex features.
# -------------------------------------------------------------------------
model.add(Conv2D(64,(3,3),activation='relu'))

# Second Pooling Layer
model.add(MaxPooling2D(pool_size=(2,2)))

# -------------------------------------------------------------------------
# Flatten Layer
#
# Converts 2D feature maps into 1D vector.
#
# Why?
# Dense layers accept only one-dimensional input.
# -------------------------------------------------------------------------
model.add(Flatten())

# -------------------------------------------------------------------------
# Fully Connected Layer
#
# Dense Layer:
# 128 neurons
#
# Learns high-level image features.
# -------------------------------------------------------------------------
model.add(Dense(128,activation='relu'))

# -------------------------------------------------------------------------
# Output Layer
#
# 10 neurons because MNIST has 10 classes.
#
# Softmax converts outputs into probabilities.
# Example:
#
# Digit 0 -> 0.01
# Digit 1 -> 0.02
# Digit 7 -> 0.95
#
# Highest probability becomes prediction.
# -------------------------------------------------------------------------
model.add(Dense(10,activation='softmax'))

# =============================================================================
# STEP 4: Compile Model
# =============================================================================

# optimizer='adam'
# Adam automatically updates weights efficiently.
#
# loss='sparse_categorical_crossentropy'
# Suitable because labels are integers (0-9).
#
# metrics=['accuracy']
# Displays classification accuracy.
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# =============================================================================
# STEP 5: Train Model
# =============================================================================

# epochs=5
#
# One epoch means the entire training dataset is used once.
#
# More epochs generally improve learning but may cause overfitting.
model.fit(x_train,y_train,epochs=5)

# =============================================================================
# STEP 6: Evaluate Model
# =============================================================================

# Evaluate model on unseen test images.
loss,accuracy = model.evaluate(x_test,y_test)

# Display accuracy.
print("Test Accuracy :",accuracy)

# =============================================================================
# STEP 7: Predict One Image
# =============================================================================

# Predict first image from test dataset.
prediction=model.predict(x_test[:1])

# Print predicted digit.
print("Predicted Digit :",prediction.argmax())

# Print actual digit.
print("Actual Digit :",y_test[0])

# =============================================================================
# EXPECTED OUTPUT
# =============================================================================
#
# Training Accuracy ≈ 99%
#
# Testing Accuracy ≈ 98% to 99%
#
# Example:
#
# Test Accuracy : 0.9894
# Predicted Digit : 7
# Actual Digit : 7
#
# =============================================================================

# =============================================================================
# VIVA QUESTIONS (WITH ANSWERS)
# =============================================================================
#
# Q1. What is MNIST?
# Ans: A handwritten digit dataset containing 70,000 grayscale images of digits
#      from 0 to 9.
#
# Q2. Why do we normalize images?
# Ans: To scale pixel values between 0 and 1, improving convergence and training
#      stability.
#
# Q3. Why reshape the images?
# Ans: Conv2D expects input in the form (samples, height, width, channels).
#
# Q4. What is Conv2D?
# Ans: A convolution layer that extracts image features using learnable filters.
#
# Q5. What is a filter (kernel)?
# Ans: A small matrix that slides across the image to detect features such as
#      edges, curves, and textures.
#
# Q6. Why use ReLU?
# Ans: It introduces non-linearity and helps avoid the vanishing gradient problem.
#
# Q7. What is MaxPooling?
# Ans: A layer that reduces image dimensions while preserving the most important
#      features.
#
# Q8. Why use Flatten?
# Ans: It converts 2D feature maps into a 1D vector for Dense layers.
#
# Q9. What is a Dense layer?
# Ans: A fully connected layer where every neuron connects to all neurons in the
#      previous layer for final classification.
#
# Q10. Why Softmax in the output layer?
# Ans: It converts outputs into probabilities whose sum equals 1.
#
# Q11. Why use Adam optimizer?
# Ans: It adapts the learning rate automatically and converges faster than
#      standard gradient descent.
#
# Q12. What is an epoch?
# Ans: One complete pass through the entire training dataset.
#
# Q13. What is batch size?
# Ans: The number of training samples processed before updating model weights.
#
# Q14. Difference between ANN and CNN?
# Ans: ANN uses fully connected layers, while CNN uses convolutional layers for
#      automatic feature extraction from images.
#
# Q15. Name some applications of CNN.
# Ans: Face recognition, OCR, medical image analysis, autonomous driving,
#      satellite image analysis, and object detection.
#
# =============================================================================