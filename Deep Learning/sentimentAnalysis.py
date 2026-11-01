# Sentiment Analysis is an NLP technique used to classify text
# as positive or negative.
# This program uses an RNN with an LSTM layer to analyse
# movie reviews from the IMDB dataset.
# LSTM remembers important information from previous words
# and helps understand the context of sentences.
# The model predicts whether a movie review is Positive or Negative.

# Step 1: Start the program.
# Step 2: Import the required libraries and IMDB dataset.
# Step 3: Load the dataset containing movie reviews.
# Step 4: Apply padding to make all sequences equal in length.
# Step 5: Build the RNN model using Embedding and LSTM layers.
# Step 6: Compile the model using Adam optimizer and binary crossentropy.
# Step 7: Train the model using the training dataset.
# Step 8: Evaluate the model using the testing dataset.
# Step 9: Predict whether a review is Positive or Negative.
# Step 10: Stop the program.

# Import required libraries
import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Load IMDB dataset
max_features = 10000
maxlen = 200

(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=max_features
)

# Apply padding
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

# Build RNN model with LSTM
model = Sequential([
    Embedding(max_features, 128),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')
])

# Compile model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(x_train, y_train, epochs=5, batch_size=32,
          validation_data=(x_test, y_test))

# Evaluate model
score, accuracy = model.evaluate(x_test, y_test)

print("Test Score:", score)
print("Test Accuracy:", accuracy)

# IMDB Dataset : Contains movie reviews labelled as positive or negative.
# RNN          : Processes sequential data.
# LSTM         : Remembers important information from previous inputs.
# Embedding    : Converts words into numerical vector representations.
# Padding      : Makes all sequences equal in length.
# Sigmoid      : Produces an output between 0 and 1.
# Adam         : Optimizer used to update model weights.
# Binary Crossentropy : Loss function used for binary classification.


# Output:
# Test Score: (Loss value)
# Test Accuracy: (Accuracy value)

# Example:
# Prediction: 0.8765
# Sentiment: Positive

# Prediction: 0.1245
# Sentiment: Negative



# Thus, a sentiment analysis model was successfully trained
# using RNN with LSTM on the IMDB dataset to classify
# movie reviews as positive or negative.

