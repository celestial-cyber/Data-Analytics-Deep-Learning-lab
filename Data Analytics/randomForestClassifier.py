# ==============================
# WEEK 5: CLASSIFICATION CONCEPTS
# EXPERIMENT 6: IMPLEMENT RANDOM FOREST CLASSIFIER
# ==============================

# Step 1: Import necessary libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# ---------------------------------------------------
# CODE EXPLANATION
# ---------------------------------------------------

# Why do we import pandas?
# Pandas is used to create and manipulate tabular data
# in the form of DataFrames.

# Why do we import RandomForestClassifier?
# RandomForestClassifier is a supervised Machine Learning
# algorithm used for classification problems.
# It combines multiple Decision Trees to improve prediction
# accuracy and reduce overfitting.

# Step 2: Create sample dataset
data = {
    'Age': [22, 25, 47, 52, 46, 56, 28, 35],
    'Income': [25000, 30000, 60000, 80000, 65000, 90000, 32000, 45000],
    'Buy': [0, 0, 1, 1, 1, 1, 0, 1]
}

# Why are we using Age and Income?
# Age and Income are the independent variables (features).
# Buy is the dependent variable (target).
# 0 means Customer will NOT Buy.
# 1 means Customer WILL Buy.

# Step 3: Convert dictionary into DataFrame
df = pd.DataFrame(data)

# Why convert to DataFrame?
# DataFrame organizes data into rows and columns,
# making it compatible with Machine Learning libraries.

# Step 4: Separate input (X) and output (y)
X = df[['Age', 'Income']]
y = df['Buy']

# Why use double brackets for X?
# X contains multiple input features and must be
# represented as a 2-dimensional DataFrame.

# Why use single brackets for y?
# y contains only one target column and is represented
# as a 1-dimensional Series.

# Step 5: Create Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Why create the model?
# This initializes the Random Forest classifier.

# Why n_estimators=100?
# It creates 100 Decision Trees.
# More trees generally improve prediction accuracy.

# Why random_state=42?
# It ensures the same random results every time
# the program is executed.

# Step 6: Train the model
model.fit(X, y)

# What does fit() do?
# fit() trains all Decision Trees using random samples
# of the dataset.
# Each tree learns different decision rules.

# Step 7: Predict for a new customer
prediction = model.predict([[40, 50000]])

# Why use predict()?
# predict() asks every Decision Tree for a prediction.
# The final prediction is decided using majority voting.

# Step 8: Display prediction
print("Prediction:", prediction[0])

# Why prediction[0]?
# predict() returns an array.
# prediction[0] extracts the first prediction.

# ---------------------------------------------------
# SAMPLE OUTPUT
# ---------------------------------------------------

# Prediction: 1

# Interpretation:
# 1 → Customer Will Buy
# 0 → Customer Will Not Buy

# Note:
# The prediction may vary slightly depending on
# the training data and random sampling.