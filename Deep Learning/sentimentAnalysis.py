# =============================================================================
#                    EXPERIMENT NO: 6
# =============================================================================
# TITLE:
# Train a Sentiment Analysis Model on the IMDB Dataset
# using Recurrent Neural Network (RNN) with LSTM
# =============================================================================

# =============================================================================
# AIM:
# =============================================================================
# To build and train a Deep Learning model using an Embedding Layer
# and Long Short-Term Memory (LSTM) network to classify movie reviews
# from the IMDB dataset as Positive or Negative.
#
# =============================================================================
# THEORY:
# =============================================================================
#
# What is Sentiment Analysis?
# ---------------------------
# Sentiment Analysis is a Natural Language Processing (NLP) technique
# used to determine whether a piece of text expresses a positive,
# negative, or neutral opinion.
#
# Examples:
# "This movie is amazing!"  --> Positive
# "Worst movie ever."       --> Negative
#
# =============================================================================
# What is IMDB Dataset?
# =============================================================================
#
# IMDB (Internet Movie Database) is a benchmark dataset for
# sentiment analysis.
#
# It contains:
#
# • 50,000 movie reviews
# • 25,000 reviews for training
# • 25,000 reviews for testing
#
# Labels:
# 1 → Positive Review
# 0 → Negative Review
#
# =============================================================================
# Why can't we use CNN here?
# =============================================================================
#
# CNN works exceptionally well for images because nearby pixels contain
# spatial information.
#
# Text is sequential in nature.
#
# Example:
#
# "I do NOT like this movie."
#
# Here, the word "NOT" changes the meaning of the sentence.
#
# CNN may miss such long-distance relationships.
#
# Therefore, sequence models like RNN, LSTM, and GRU are preferred.
#
# =============================================================================
# What is RNN?
# =============================================================================
#
# RNN stands for Recurrent Neural Network.
#
# Unlike ANN, RNN remembers previous information.
#
# Example:
#
# Word1 → Word2 → Word3 → Word4
#        ↑        ↑
# Previous information is carried forward.
#
# Hence, RNN is useful for:
#
# • Text
# • Speech
# • Time Series
# • Language Translation
#
# =============================================================================
# Problem with Simple RNN
# =============================================================================
#
# RNN suffers from the Vanishing Gradient Problem.
#
# When sequences become long,
# earlier information gradually disappears.
#
# Example:
#
# "I watched the movie yesterday because my friend
# recommended it and I absolutely loved it."
#
# A simple RNN may forget the beginning of the sentence.
#
# =============================================================================
# Solution → LSTM
# =============================================================================
#
# LSTM = Long Short-Term Memory
#
# LSTM solves the Vanishing Gradient Problem.
#
# It remembers important information for a long time
# and forgets unnecessary information.
#
# LSTM contains three gates:
#
# 1. Forget Gate
#    Removes unnecessary information.
#
# 2. Input Gate
#    Stores useful information.
#
# 3. Output Gate
#    Produces the final output.
#
# Therefore,
# LSTM performs much better than Simple RNN on long text sequences.
#
# =============================================================================
# What is GRU?
# =============================================================================
#
# GRU = Gated Recurrent Unit
#
# GRU is a simplified version of LSTM.
#
# Advantages:
#
# • Faster training
# • Fewer parameters
# • Similar accuracy
#
# LSTM has:
#   Forget Gate
#   Input Gate
#   Output Gate
#
# GRU has:
#   Update Gate
#   Reset Gate
#
# =============================================================================
# Workflow
# =============================================================================
#
# Movie Review
#        ↓
# Tokenization
#        ↓
# Integer Encoding
#        ↓
# Padding
#        ↓
# Embedding Layer
#        ↓
# LSTM Layer
#        ↓
# Dense Layer
#        ↓
# Positive / Negative Prediction
#
# =============================================================================

# =============================================================================
# IMPORT LIBRARIES
# =============================================================================

# TensorFlow is Google's Deep Learning Framework.
# It provides ready-made tools to build neural networks.
import tensorflow as tf

# Import IMDB Dataset.
# This dataset is already available inside TensorFlow.
from tensorflow.keras.datasets import imdb

# Pad Sequences
#
# Reviews have different lengths.
#
# Example:
#
# Review 1 -> 120 words
# Review 2 -> 40 words
# Review 3 -> 300 words
#
# Neural Networks require equal-sized inputs.
#
# pad_sequences() makes every review the same length.
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sequential Model
#
# Allows layers to be added one after another.
from tensorflow.keras.models import Sequential

# Import Required Layers
#
# Embedding
# LSTM
# Dense
from tensorflow.keras.layers import Embedding, LSTM, Dense

# =============================================================================
# STEP 1 : LOAD IMDB DATASET
# =============================================================================

# num_words=10000
#
# Keeps only the 10,000 most frequent words.
#
# Rare words are ignored.
#
# Why?
#
# Reduces memory usage
# Speeds up training
# Removes unnecessary vocabulary.
#
# Returns:
#
# x_train -> Training Reviews
# y_train -> Training Labels
#
# x_test -> Testing Reviews
# y_test -> Testing Labels
#
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)

# =============================================================================
# Display Dataset Information
# =============================================================================

print("Training Samples :", len(x_train))
print("Testing Samples  :", len(x_test))

# =============================================================================
# STEP 2 : Padding
# =============================================================================

# Reviews have different lengths.
#
# Example:
#
# Review 1
# [12,56,77,45]
#
# Review 2
# [11,2]
#
# Review 3
# [4,5,6,7,8,9,10]
#
# pad_sequences() converts them into equal length.
#
# maxlen=200
#
# Every review becomes exactly 200 words.
#
# Longer reviews are truncated.
#
# Shorter reviews are padded with zeros.
#
x_train = pad_sequences(x_train, maxlen=200)

x_test = pad_sequences(x_test, maxlen=200)

# =============================================================================
# STEP 3 : BUILD THE MODEL
# =============================================================================

# Create Sequential Model
model = Sequential()

# =============================================================================
# Embedding Layer
# =============================================================================
#
# Embedding converts every word into a dense vector.
#
# Example:
#
# "good"
# ↓
# [0.12, 0.45, -0.88, ...]
#
# Words with similar meanings get similar vectors.
#
# input_dim = Vocabulary Size
#
# output_dim = Embedding Dimension
#
# input_length = Length of Review
#
model.add(
    Embedding(
        input_dim=10000,
        output_dim=128,
        input_length=200
    )
)

# =============================================================================
# LSTM Layer
# =============================================================================
#
# 128 neurons.
#
# Learns long-term dependencies in text.
#
# Remembers important words.
#
# Example:
#
# "Although the movie started slowly,
# the ending was fantastic."
#
# LSTM understands that "fantastic"
# makes the sentiment positive.
#
model.add(
    LSTM(128)
)

# =============================================================================
# Dense Layer
# =============================================================================
#
# Only one neuron.
#
# Because:
#
# Output is either
#
# Positive
#
# OR
#
# Negative
#
# Sigmoid Activation
#
# Output Range:
#
# 0 → Negative
#
# 1 → Positive
#
model.add(
    Dense(
        1,
        activation='sigmoid'
    )
)

# =============================================================================
# Display Model Summary
# =============================================================================

# Shows:
#
# Number of layers
# Parameters
# Output Shapes
#
model.summary()

# =============================================================================
# END OF PART 1
# =============================================================================
#
