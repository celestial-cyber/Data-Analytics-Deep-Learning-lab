# =============================================================================
# EXPERIMENT NO: 5
# TITLE: Applying Deep Learning Models in the field of
# Natural Language Processing (NLP)
# =============================================================================

# AIM:
# To implement a Deep Learning model using an Embedding layer and LSTM
# for sentiment classification on text data.

# =============================================================================
# THEORY:
#
# What is NLP?
# ------------
# Natural Language Processing (NLP) is a branch of Artificial Intelligence
# that enables computers to understand, process, analyze, and generate
# human language.
#
# Examples of NLP:
# - Chatbots (ChatGPT)
# - Google Translate
# - Speech Recognition
# - Spam Email Detection
# - Sentiment Analysis
# - Text Summarization
#
# Why Deep Learning in NLP?
# -------------------------
# Traditional Machine Learning requires manual feature extraction.
# Deep Learning automatically learns meaningful representations
# (features) from text.
#
# This experiment uses:
#
# Text
#   ↓
# Tokenization
#   ↓
# Padding
#   ↓
# Embedding Layer
#   ↓
# LSTM Layer
#   ↓
# Dense Layer
#   ↓
# Sentiment Prediction
#
# =============================================================================

# Import TensorFlow library.
# TensorFlow is used to build and train Deep Learning models.
import tensorflow as tf

# Import Sequential model.
from tensorflow.keras.models import Sequential

# Import required layers.
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Import Tokenizer and Padding.
# Tokenizer converts words into numbers.
# Padding makes every sentence the same length.
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =============================================================================
# STEP 1 : Create Sample Dataset
# =============================================================================

# Positive and negative movie reviews.
texts = [
    "I love this movie",
    "This movie is amazing",
    "Fantastic acting",
    "I enjoyed the film",
    "Very good movie",
    "I hate this movie",
    "Worst film ever",
    "Terrible acting",
    "Very boring movie",
    "I dislike this film"
]

# Labels:
# 1 = Positive
# 0 = Negative
labels = [1,1,1,1,1,0,0,0,0,0]

# =============================================================================
# STEP 2 : Tokenization
# =============================================================================

# Create tokenizer object.
tokenizer = Tokenizer()

# Build vocabulary from text.
tokenizer.fit_on_texts(texts)

# Convert words into integer sequences.
sequences = tokenizer.texts_to_sequences(texts)

# =============================================================================
# STEP 3 : Padding
# =============================================================================

# Make every sentence length equal.
# Short sentences receive zeros at the beginning.
data = pad_sequences(sequences,maxlen=5)

# =============================================================================
# STEP 4 : Build Deep Learning Model
# =============================================================================

# Create Sequential Model.
model = Sequential()

# --------------------------------------------------------------------
# Embedding Layer
#
# input_dim = Vocabulary size
#
# output_dim = Embedding vector size
#
# input_length = Length of each sentence
#
# Purpose:
# Converts word IDs into dense vectors that capture semantic meaning.
# --------------------------------------------------------------------
model.add(Embedding(input_dim=100,
                    output_dim=16,
                    input_length=5))

# --------------------------------------------------------------------
# LSTM Layer
#
# LSTM (Long Short-Term Memory)
#
# Purpose:
# Learns relationships between words while remembering long-term context.
# --------------------------------------------------------------------
model.add(LSTM(32))

# --------------------------------------------------------------------
# Dense Layer
#
# Sigmoid is used because this is a binary classification problem.
#
# Output:
# 0 → Negative
# 1 → Positive
# --------------------------------------------------------------------
model.add(Dense(1,activation='sigmoid'))

# =============================================================================
# STEP 5 : Compile Model
# =============================================================================

# Adam optimizer updates weights efficiently.
#
# Binary Crossentropy is used for binary classification.
#
# Accuracy displays prediction performance.
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# =============================================================================
# STEP 6 : Train Model
# =============================================================================

# Train the model for 20 epochs.
model.fit(data,labels,epochs=20,verbose=1)

# =============================================================================
# STEP 7 : Test New Sentence
# =============================================================================

# Example sentence.
test_sentence=["I really love this film"]

# Convert into sequence.
test_seq=tokenizer.texts_to_sequences(test_sentence)

# Pad to same length.
test_pad=pad_sequences(test_seq,maxlen=5)

# Predict sentiment.
prediction=model.predict(test_pad)

# Display result.
if prediction[0][0] > 0.5:
    print("Positive Review")
else:
    print("Negative Review")

# =============================================================================
# EXPECTED OUTPUT
# =============================================================================
#
# Epoch 20/20
# Accuracy : 100% (small dataset)
#
# Prediction:
#
# Positive Review
#
# =============================================================================

# =============================================================================
# APPLICATIONS OF NLP
# =============================================================================
#
# 1. Chatbots
# 2. Voice Assistants
# 3. Machine Translation
# 4. Sentiment Analysis
# 5. Spam Detection
# 6. Text Summarization
# 7. Question Answering
# 8. Email Classification
# 9. Search Engines
# 10. Recommendation Systems
#
# =============================================================================

# =============================================================================
# ADVANTAGES
# =============================================================================
#
# • Automatically learns language patterns.
# • Handles large text datasets efficiently.
# • Produces high accuracy.
# • Reduces manual feature engineering.
#
# =============================================================================

# =============================================================================
# LIMITATIONS
# =============================================================================
#
# • Requires large datasets for best performance.
# • Computationally expensive.
# • Training can take longer than traditional ML.
#
# =============================================================================

# =============================================================================
# VIVA QUESTIONS (WITH ANSWERS)
# =============================================================================
#
# Q1. What is NLP?
# Ans: NLP (Natural Language Processing) enables computers to understand,
#      analyze, and generate human language.
#
# Q2. Give some applications of NLP.
# Ans: Chatbots, translation, spam detection, sentiment analysis,
#      speech recognition, and text summarization.
#
# Q3. Why is tokenization required?
# Ans: Neural networks work with numbers, so tokenization converts words into
#      integer IDs.
#
# Q4. What is padding?
# Ans: Padding makes all input sequences the same length by adding zeros.
#
# Q5. What is an embedding layer?
# Ans: It converts word IDs into dense vectors that represent semantic meaning.
#
# Q6. What is LSTM?
# Ans: Long Short-Term Memory is a type of Recurrent Neural Network (RNN) that
#      remembers long-term dependencies in sequences.
#
# Q7. Why use LSTM instead of a simple RNN?
# Ans: LSTM reduces the vanishing gradient problem and captures long-range
#      dependencies more effectively.
#
# Q8. Why use sigmoid activation?
# Ans: Sigmoid outputs values between 0 and 1, making it suitable for binary
#      classification.
#
# Q9. Why use binary_crossentropy?
# Ans: It is the appropriate loss function for binary classification tasks.
#
# Q10. What optimizer is used?
# Ans: Adam optimizer.
#
# Q11. What is sentiment analysis?
# Ans: Identifying whether text expresses a positive, negative, or neutral
#      opinion.
#
# Q12. Difference between NLP and Computer Vision?
# Ans: NLP processes text and language, whereas Computer Vision processes images
#      and videos.
#
# Q13. What is a vocabulary in NLP?
# Ans: The set of unique words learned by the tokenizer.
#
# Q14. What is sequence length?
# Ans: The number of tokens in a sentence after tokenization and padding.
#
# Q15. Mention real-world applications of Deep Learning in NLP.
# Ans: ChatGPT, Google Translate, Siri, Alexa, Grammarly, spam filters,
#      search engines, and customer support chatbots.
#
# =============================================================================