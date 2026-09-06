# ============================================================
# IMPORT LIBRARIES
# ============================================================

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Google Colab file upload
from google.colab import files


# ============================================================
# SET PARAMETERS
# ============================================================

plt.rc("figure", autolayout=True)
plt.rc("image", cmap="magma")


# ============================================================
# UPLOAD IMAGE
# ============================================================

print("Please upload an image (JPG/JPEG/PNG)...")

uploaded = files.upload()

# Get the uploaded filename automatically
filename = next(iter(uploaded))

print("Uploaded file:", filename)


# ============================================================
# DEFINE THE KERNEL
# ============================================================

# Edge detection / sharpening kernel

kernel = tf.constant([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
], dtype=tf.float32)


# ============================================================
# LOAD THE IMAGE
# ============================================================

image_path = filename

image = tf.io.read_file(image_path)


# ============================================================
# DECODE IMAGE
# ============================================================

# Convert image to grayscale

image = tf.io.decode_image(
    image,
    channels=1,
    expand_animations=False
)


# ============================================================
# RESIZE IMAGE
# ============================================================

image = tf.image.resize(
    image,
    size=[300, 300]
)


# ============================================================
# DISPLAY ORIGINAL IMAGE
# ============================================================

img = tf.squeeze(image).numpy()

plt.figure(figsize=(5, 5))

plt.imshow(
    img,
    cmap="gray"
)

plt.axis("off")

plt.title("Original Gray Scale Image")

plt.show()


# ============================================================
# PREPROCESS IMAGE
# ============================================================

# Convert pixel values from 0-255 to 0-1

image = tf.image.convert_image_dtype(
    image,
    dtype=tf.float32
)


# Add batch dimension
#
# Before:
# (300, 300, 1)
#
# After:
# (1, 300, 300, 1)

image = tf.expand_dims(
    image,
    axis=0
)


# ============================================================
# PREPARE KERNEL
# ============================================================

# Conv2D expects:
#
# (height, width, input_channels, output_channels)
#
# Therefore:
#
# (3, 3)
# becomes
# (3, 3, 1, 1)

kernel = tf.reshape(
    kernel,
    [3, 3, 1, 1]
)


# ============================================================
# CONVOLUTIONAL LAYER
# ============================================================

image_filter = tf.nn.conv2d(
    input=image,
    filters=kernel,
    strides=[1, 1, 1, 1],
    padding="SAME"
)


# ============================================================
# CREATE RESULT FIGURE
# ============================================================

plt.figure(figsize=(15, 5))


# ============================================================
# CONVOLUTION RESULT
# ============================================================

plt.subplot(1, 3, 1)

plt.imshow(
    tf.squeeze(image_filter).numpy(),
    cmap="gray"
)

plt.axis("off")

plt.title("Convolution")


# ============================================================
# ACTIVATION LAYER - ReLU
# ============================================================

image_detect = tf.nn.relu(
    image_filter
)


plt.subplot(1, 3, 2)

plt.imshow(
    tf.squeeze(image_detect).numpy(),
    cmap="gray"
)

plt.axis("off")

plt.title("Activation - ReLU")


# ============================================================
# POOLING LAYER - MAX POOLING
# ============================================================

image_condense = tf.nn.max_pool2d(
    input=image_detect,
    ksize=2,
    strides=2,
    padding="SAME"
)


# ============================================================
# DISPLAY POOLING RESULT
# ============================================================

plt.subplot(1, 3, 3)

plt.imshow(
    tf.squeeze(image_condense).numpy(),
    cmap="gray"
)

plt.axis("off")

plt.title("Max Pooling")


# ============================================================
# DISPLAY ALL RESULTS
# ============================================================

plt.show()


# ============================================================
# DISPLAY SHAPES
# ============================================================

print("==========================================")
print("IMAGE PROCESSING COMPLETE")
print("==========================================")

print("Original image shape:",
      image.shape)

print("Convolution output shape:",
      image_filter.shape)

print("ReLU output shape:",
      image_detect.shape)

print("Pooling output shape:",
      image_condense.shape)





      # ============================================================
# OBJECTIVE
# ============================================================
# To demonstrate the basic working of a Convolutional Neural
# Network (CNN) by applying three important operations:
#
# 1. Convolution
# 2. ReLU Activation
# 3. Max Pooling
#
# The program takes an input image, extracts important visual
# features such as edges, removes negative activations, and
# reduces the size of the feature map.


# ============================================================
# STEP 1: IMPORT LIBRARIES
# ============================================================
# NumPy      -> Numerical operations
# TensorFlow  -> Image processing and CNN operations
# Matplotlib  -> Display the images and feature maps
# Colab files -> Upload an image from the local computer


# ============================================================
# STEP 2: UPLOAD IMAGE
# ============================================================
# Upload an image from the user's computer.
# The uploaded filename is automatically detected so the user
# does not need to rename the image to a specific name.


# ============================================================
# STEP 3: DEFINE CONVOLUTION KERNEL
# ============================================================
# A 3x3 edge-detection kernel is created.
#
# The kernel moves over small regions of the image and performs
# mathematical calculations to detect sudden changes in pixel
# intensity.
#
# This helps identify edges and boundaries in the image.


# ============================================================
# STEP 4: LOAD AND DECODE IMAGE
# ============================================================
# The uploaded image is read using TensorFlow.
#
# The image is converted into grayscale by using one channel.
# This simplifies the image because we only consider pixel
# intensity instead of Red, Green and Blue color information.


# ============================================================
# STEP 5: RESIZE IMAGE
# ============================================================
# The image is resized to 300 x 300 pixels.
#
# A fixed size makes the input consistent for further
# convolution operations.


# ============================================================
# STEP 6: DISPLAY ORIGINAL IMAGE
# ============================================================
# The original grayscale image is displayed so that we can
# compare it with the feature maps generated later.


# ============================================================
# STEP 7: IMAGE PREPROCESSING
# ============================================================
# Pixel values normally range from 0 to 255.
#
# They are converted to floating-point values between 0 and 1.
#
# A batch dimension is also added because TensorFlow's
# convolution operation expects the input in the format:
#
# (batch, height, width, channels)


# ============================================================
# STEP 8: PREPARE THE KERNEL
# ============================================================
# The 3x3 kernel is reshaped into:
#
# (height, width, input_channels, output_channels)
#
# In our case:
#
# (3, 3, 1, 1)
#
# This makes the kernel compatible with TensorFlow's Conv2D.


# ============================================================
# STEP 9: CONVOLUTION
# ============================================================
# The kernel slides across the image.
#
# At every position, it performs a mathematical operation
# between the kernel values and the corresponding image pixels.
#
# The resulting feature map highlights important features,
# especially edges and boundaries.


# ============================================================
# STEP 10: RELU ACTIVATION
# ============================================================
# ReLU stands for Rectified Linear Unit.
#
# ReLU(x) = max(0, x)
#
# Negative values are converted to zero while positive values
# are retained.
#
# This introduces non-linearity and keeps useful activations.


# ============================================================
# STEP 11: MAX POOLING
# ============================================================
# Max Pooling reduces the size of the feature map.
#
# For every 2x2 region, the maximum value is selected.
#
# This keeps the strongest detected feature while reducing
# the amount of data and computational cost.


# ============================================================
# STEP 12: VISUALIZATION
# ============================================================
# Finally, we display:
#
# 1. Convolution output
# 2. ReLU activation output
# 3. Max pooling output
#
# This allows us to visually understand how an image is
# transformed as it passes through the basic stages of a CNN.


# ============================================================
# FINAL OBJECTIVE
# ============================================================
# The main objective is to understand how CNNs extract
# important visual features from images using:
#
#        Image
#          ↓
#     Convolution
#          ↓
#        ReLU
#          ↓
#     Max Pooling
#          ↓
#    Feature Extraction
#
# This program demonstrates the fundamental building blocks
# used in image-based deep learning and computer vision.