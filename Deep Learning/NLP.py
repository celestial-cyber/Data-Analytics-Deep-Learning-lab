# Natural Language Processing (NLP) enables computers to understand human language.
# This program uses a Deep Neural Network (DNN) to perform emotion classification.
# It uses Tokenization to convert words into numbers.
# Embedding converts words into meaningful numerical representations.
# The trained model predicts one of six emotions:
# Sadness, Joy, Love, Anger, Surprise, and Fear.

# Step 1: Start the program.
# Step 2: Import the required libraries and load the emotion dataset.
# Step 3: Tokenize text and convert words into numerical sequences.
# Step 4: Apply padding to make all sequences equal in length.
# Step 5: Encode emotion labels into numerical values.
# Step 6: Split the dataset into training and testing sets.
# Step 7: Build a DNN model using Embedding, Flatten and Dense layers.
# Step 8: Compile and train the model.
# Step 9: Accept a sentence from the user.
# Step 10: Predict and display the emotion.
# Step 11: Repeat until the user enters 'exit'.
# Step 12: Stop the program.


# Import required libraries
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Load dataset
data = pd.read_csv("train.txt", sep=";", header=None,
                   names=["Text", "Emotions"])

print(data.head())

# Convert text into sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data.Text)
X = tokenizer.texts_to_sequences(data.Text)

# Apply padding
max_len = max(len(i) for i in X)
X = pad_sequences(X, maxlen=max_len)

# Encode emotion labels
encoder = LabelEncoder()
y = to_categorical(encoder.fit_transform(data.Emotions))

# Split dataset
xtrain, xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2
)

# Build and train DNN model
model = Sequential([
    Embedding(len(tokenizer.word_index)+1, 128),
    Flatten(),
    Dense(128, activation="relu"),
    Dense(y.shape[1], activation="softmax")
])

model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

model.fit(xtrain, ytrain, epochs=10, batch_size=32,
          validation_data=(xtest, ytest))

# Predict emotions from user input
while True:
    text = input("Enter a sentence (or type 'exit' to quit): ")

    if text.lower() == "exit":
        break

    seq = tokenizer.texts_to_sequences([text])
    seq = pad_sequences(seq, maxlen=max_len)

    pred = model.predict(seq, verbose=0)
    print("Predicted Emotion:", encoder.inverse_transform(
        [np.argmax(pred)]
    )[0])



    # Sample input:
# She was exhausted last night

# Output:
# Predicted Emotion: Sadness

# Sample input:
# I will feel happy

# Output:
# Predicted Emotion: Joy

# Sample input:
# He met with an accident

# Output:
# Predicted Emotion: Fear

# Note: Actual predictions depend on the trained model and dataset.
# Enter 'exit' to terminate the program.

# Tokenization  : Converts text into numerical sequences.
# Padding       : Makes all input sequences equal in length.
# Embedding     : Converts words into dense vector representations.
# Flatten       : Converts multidimensional data into a single vector.
# Dense         : Performs learning using neural network layers.
# Softmax       : Predicts the probability of each emotion.
# Adam          : Optimizer used to update model weights.


# Thus, a Deep Neural Network was implemented successfully
# to classify emotions from text using NLP.


