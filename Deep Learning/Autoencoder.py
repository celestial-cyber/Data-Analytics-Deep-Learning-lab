# =============================================================================
#                        EXPERIMENT NO: 7
# =============================================================================
#
# TITLE:
#
# Applying the Autoencoder Algorithm on the
# KDD Cup 1999 Dataset for Anomaly Detection
#
# =============================================================================
#
# AIM:
#
# To implement an Autoencoder using TensorFlow/Keras for encoding the
# KDD Cup 1999 dataset and detecting anomalies based on reconstruction error.
#
# =============================================================================
# THEORY
# =============================================================================
#
# What is an Autoencoder?
#
# An Autoencoder is an Unsupervised Deep Learning algorithm.
#
# It learns how to compress data into a smaller representation
# and reconstruct it back.
#
# Input
#   ↓
# Encoder
#   ↓
# Latent Space (Compressed Representation)
#   ↓
# Decoder
#   ↓
# Reconstructed Output
#
# Goal:
#
# Input ≈ Output
#
# =============================================================================
#
# Why Autoencoder?
#
# Instead of classifying data,
# Autoencoders learn hidden patterns.
#
# They are mainly used for:
#
# • Dimensionality Reduction
# • Noise Removal
# • Feature Extraction
# • Image Compression
# • Fraud Detection
# • Intrusion Detection
#
# =============================================================================
#
# What is KDD Cup 1999 Dataset?
#
# The KDD Cup 1999 dataset is one of the most famous
# cybersecurity datasets.
#
# It contains:
#
# • Normal Network Traffic
# • Attack Traffic
#
# Attack Types:
#
# • DoS
# • Probe
# • R2L
# • U2R
#
# Applications:
#
# Intrusion Detection Systems (IDS)
#
# =============================================================================
#
# Why Autoencoder for KDD Dataset?
#
# Autoencoder is trained only on
# NORMAL traffic.
#
# During testing:
#
# Normal Traffic
# ↓
# Small Reconstruction Error
#
# Attack Traffic
# ↓
# Large Reconstruction Error
#
# Hence,
# attacks can be detected without explicit labels.
#
# =============================================================================
#
# IMPORT LIBRARIES
# =============================================================================

import pandas as pd

# Pandas
# Used for loading and processing datasets.

import numpy as np

# NumPy
# Used for numerical computations.

from sklearn.datasets import fetch_kddcup99

# Downloads the KDD Cup 1999 dataset.

from sklearn.preprocessing import StandardScaler

# StandardScaler normalizes features
# to mean = 0 and standard deviation = 1.

from sklearn.model_selection import train_test_split

# Splits dataset into training and testing sets.

import tensorflow as tf

# TensorFlow provides deep learning APIs.

from tensorflow.keras.models import Model

# Functional API model.

from tensorflow.keras.layers import Input, Dense

# Dense Layers used for Encoder and Decoder.

# =============================================================================
# STEP 1 : LOAD DATASET
# =============================================================================

# percent10=True
#
# Downloads only 10% of the dataset.
#
# Faster for laboratory experiments.

kdd = fetch_kddcup99(percent10=True)

# Feature Matrix

X = pd.DataFrame(kdd.data)

# Labels

y = pd.Series(kdd.target)

print("Dataset Shape :", X.shape)

# =============================================================================
# STEP 2 : PREPROCESSING
# =============================================================================
#
# KDD dataset contains
# categorical and numerical columns.
#
# We first convert categorical columns
# into numerical values.

X = pd.get_dummies(X)

# =============================================================================
# NORMALIZATION
# =============================================================================
#
# Neural Networks train faster
# when every feature has similar scale.
#
# StandardScaler converts
#
# Mean = 0
#
# Standard Deviation = 1

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# =============================================================================
# TRAIN TEST SPLIT
# =============================================================================
#
# 80% Training
#
# 20% Testing

X_train, X_test = train_test_split(

    X_scaled,

    test_size=0.2,

    random_state=42

)

print("Training Samples :", X_train.shape)

print("Testing Samples :", X_test.shape)

# =============================================================================
# STEP 3 : BUILD AUTOENCODER
# =============================================================================

# Number of input features.

input_dim = X_train.shape[1]

# Input Layer

input_layer = Input(shape=(input_dim,))

# =============================================================================
# ENCODER
# =============================================================================
#
# Encoder compresses the data.

encoded = Dense(

    64,

    activation='relu'

)(input_layer)

encoded = Dense(

    32,

    activation='relu'

)(encoded)

encoded = Dense(

    16,

    activation='relu'

)(encoded)

# =============================================================================
# LATENT SPACE
# =============================================================================
#
# 16 neurons represent
# compressed information.

# =============================================================================
# DECODER
# =============================================================================
#
# Decoder reconstructs
# the original input.

decoded = Dense(

    32,

    activation='relu'

)(encoded)

decoded = Dense(

    64,

    activation='relu'

)(decoded)

decoded = Dense(

    input_dim,

    activation='linear'

)(decoded)

# =============================================================================
# CREATE MODEL
# =============================================================================

autoencoder = Model(

    inputs=input_layer,

    outputs=decoded

)

# =============================================================================
# DISPLAY MODEL SUMMARY
# =============================================================================

autoencoder.summary()

# =============================================================================
# END OF PART 1
# =============================================================================
# =============================================================================
# STEP 4 : COMPILE THE AUTOENCODER
# =============================================================================
#
# Before training, the model must be compiled.
#
# Compiling specifies:
# 1. Optimizer
# 2. Loss Function
#
# Optimizer:
# Adam (Adaptive Moment Estimation)
#
# Why Adam?
# • Fast convergence
# • Automatically adjusts learning rate
# • Most commonly used optimizer in Deep Learning
#
# Loss Function:
# Mean Squared Error (MSE)
#
# Why MSE?
#
# Autoencoders try to reconstruct the input.
#
# Input:
# [1.2, 0.8, 5.1]
#
# Output:
# [1.1, 0.9, 5.0]
#
# MSE measures how close the reconstructed output
# is to the original input.
#
# Smaller MSE = Better reconstruction.
#
# =============================================================================

autoencoder.compile(

    optimizer='adam',

    loss='mse'

)

# =============================================================================
# STEP 5 : TRAIN THE MODEL
# =============================================================================
#
# Autoencoder is trained using:
#
# Input  -> X_train
#
# Target -> X_train
#
# Notice:
#
# Input and Output are SAME.
#
# Unlike classification,
# we don't predict labels.
#
# We reconstruct the input itself.
#
# epochs = Number of complete passes through dataset.
#
# batch_size = Number of samples processed together.
#
# validation_split = 20%
#
# Used to monitor model performance during training.
#
# =============================================================================

history = autoencoder.fit(

    X_train,

    X_train,

    epochs=20,

    batch_size=128,

    validation_split=0.2,

    shuffle=True,

    verbose=1

)

# =============================================================================
# STEP 6 : RECONSTRUCT TEST DATA
# =============================================================================
#
# Predict reconstructed data.
#
# Autoencoder attempts to recreate
# the original input.
#
# =============================================================================

X_pred = autoencoder.predict(X_test)

# =============================================================================
# STEP 7 : CALCULATE RECONSTRUCTION ERROR
# =============================================================================
#
# Reconstruction Error =
#
# Original Data
#
#          -
#
# Reconstructed Data
#
# If reconstruction error is:
#
# Small
# → Normal Traffic
#
# Large
# → Suspicious Traffic
#
# =============================================================================

mse = np.mean(

    np.power(

        X_test - X_pred,

        2

    ),

    axis=1

)

print("\nFirst 10 Reconstruction Errors:\n")

print(mse[:10])

# =============================================================================
# STEP 8 : ANOMALY DETECTION
# =============================================================================
#
# Threshold Selection
#
# Here we choose:
#
# Mean Error
#
# +
#
# 2 × Standard Deviation
#
# Any sample having reconstruction error
# greater than threshold
# is treated as anomaly.
#
# =============================================================================

threshold = np.mean(mse) + (2 * np.std(mse))

print("\nThreshold :", threshold)

# =============================================================================
# Detect anomalies.
# =============================================================================

anomalies = mse > threshold

# =============================================================================
# Count anomalies.
# =============================================================================

print("\nTotal Samples :", len(anomalies))

print("Detected Anomalies :", np.sum(anomalies))

print("Normal Samples :", len(anomalies) - np.sum(anomalies))

# =============================================================================
# Display first few anomaly indices.
# =============================================================================

print("\nFirst 20 Anomaly Positions:")

print(np.where(anomalies)[0][:20])

# =============================================================================
# EXPECTED OUTPUT
# =============================================================================
#
# Dataset Shape :
# (494021, 121)
#
# Training Samples :
# (395216,121)
#
# Testing Samples :
# (98805,121)
#
# Epoch 20/20
#
# Loss : 0.0021
#
# Validation Loss : 0.0025
#
# Threshold :
# 0.014
#
# Total Samples :
# 98805
#
# Detected Anomalies :
# 1500 (approximately)
#
# =============================================================================
# APPLICATIONS OF AUTOENCODER
# =============================================================================
#
# 1. Intrusion Detection
#
# 2. Credit Card Fraud Detection
#
# 3. Medical Image Compression
#
# 4. Noise Removal
#
# 5. Image Compression
#
# 6. Recommendation Systems
#
# 7. Feature Extraction
#
# 8. Dimensionality Reduction
#
# 9. Manufacturing Defect Detection
#
# 10. Network Security
#
# =============================================================================
# ADVANTAGES
# =============================================================================
#
# • Learns automatically without labels.
#
# • Excellent for anomaly detection.
#
# • Reduces dimensionality.
#
# • Learns hidden representations.
#
# • Works well on large datasets.
#
# =============================================================================
# LIMITATIONS
# =============================================================================
#
# • Requires significant training data.
#
# • Threshold selection is important.
#
# • Computationally expensive.
#
# • Can reconstruct anomalies if overtrained.
#
# =============================================================================
# REAL-WORLD ANALOGY
# =============================================================================
#
# Imagine a bank cashier who has handled genuine currency notes
# for many years.
#
# When a fake note appears, the cashier immediately notices
# that it "doesn't look right."
#
# Similarly:
#
# Autoencoder learns only NORMAL patterns.
#
# When abnormal data arrives,
# reconstruction becomes poor,
# leading to a HIGH reconstruction error.
#
# Hence,
# anomaly is detected.
#
# =============================================================================
# COMMON ERRORS
# =============================================================================
#
# Error:
# ValueError due to categorical columns.
#
# Solution:
# Use pd.get_dummies() before scaling.
#
# ------------------------------------------------------------
#
# Error:
# Features not normalized.
#
# Solution:
# Apply StandardScaler().
#
# ------------------------------------------------------------
#
# Error:
# High reconstruction error for all samples.
#
# Solution:
# Increase epochs or tune network architecture.
#
# ------------------------------------------------------------
#
# Error:
# Too many anomalies detected.
#
# Solution:
# Adjust the threshold.
#
# =============================================================================
# VIVA QUESTIONS WITH ANSWERS
# =============================================================================
#
# Q1. What is an Autoencoder?
# Ans:
# An Autoencoder is an unsupervised neural network that learns to compress
# input data and reconstruct it with minimal error.
#
# ------------------------------------------------------------
#
# Q2. Is Autoencoder supervised or unsupervised?
# Ans:
# Unsupervised Learning.
#
# ------------------------------------------------------------
#
# Q3. What are the two main parts of an Autoencoder?
# Ans:
# Encoder and Decoder.
#
# ------------------------------------------------------------
#
# Q4. What is the Encoder?
# Ans:
# It compresses the input into a lower-dimensional representation.
#
# ------------------------------------------------------------
#
# Q5. What is the Decoder?
# Ans:
# It reconstructs the original input from the compressed representation.
#
# ------------------------------------------------------------
#
# Q6. What is Latent Space?
# Ans:
# A compressed representation containing the most important features.
#
# ------------------------------------------------------------
#
# Q7. Why use MSE loss?
# Ans:
# Because the model reconstructs continuous input values, and MSE measures
# reconstruction accuracy.
#
# ------------------------------------------------------------
#
# Q8. What is Reconstruction Error?
# Ans:
# The difference between the original input and the reconstructed output.
#
# ------------------------------------------------------------
#
# Q9. Why does anomaly detection work with Autoencoders?
# Ans:
# The model learns normal patterns well, so abnormal data produces a larger
# reconstruction error.
#
# ------------------------------------------------------------
#
# Q10. What is the KDD Cup 1999 dataset?
# Ans:
# A benchmark cybersecurity dataset used for intrusion detection research.
#
# ------------------------------------------------------------
#
# Q11. Name some attacks present in the KDD dataset.
# Ans:
# DoS, Probe, R2L, and U2R.
#
# ------------------------------------------------------------
#
# Q12. Why normalize data before training?
# Ans:
# To bring all features to a similar scale, improving training stability.
#
# ------------------------------------------------------------
#
# Q13. Why use StandardScaler?
# Ans:
# It standardizes features to zero mean and unit variance.
#
# ------------------------------------------------------------
#
# Q14. Can Autoencoders be used for dimensionality reduction?
# Ans:
# Yes, the encoder learns a compact representation of the data.
#
# ------------------------------------------------------------
#
# Q15. Mention applications of Autoencoders.
# Ans:
# Fraud detection, anomaly detection, image compression, denoising,
# feature extraction, and recommendation systems.
#
# ------------------------------------------------------------
#
# Q16. Difference between PCA and Autoencoder?
# Ans:
# PCA performs linear dimensionality reduction, while Autoencoders can learn
# nonlinear representations using neural networks.
#
# ------------------------------------------------------------
#
# Q17. What optimizer is used?
# Ans:
# Adam optimizer.
#
# ------------------------------------------------------------
#
# Q18. What happens if the latent space is too small?
# Ans:
# Important information may be lost, increasing reconstruction error.
#
# ------------------------------------------------------------
#
# Q19. Why is validation_split used?
# Ans:
# To monitor model performance on unseen validation data during training.
#
# ------------------------------------------------------------
#
# Q20. What is the role of epochs?
# Ans:
# Epochs determine how many times the entire training dataset is processed.
#
# =============================================================================
# END OF EXPERIMENT
# =============================================================================