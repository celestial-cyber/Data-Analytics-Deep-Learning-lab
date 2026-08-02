# ==============================
# WEEK 5: CLASSIFICATION CONCEPTS
# EXPERIMENT 5: IMPLEMENT DECISION TREE INDUCTION FOR CLASSIFICATION
# ==============================

# Step 1: Import necessary libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# ---------------------------------------------------
# CODE EXPLANATION
# ---------------------------------------------------

# Why do we import pandas?
# Pandas is used to create and manipulate tabular data
# in the form of DataFrames.

# Why do we import DecisionTreeClassifier?
# DecisionTreeClassifier is a supervised Machine Learning
# algorithm used for classification problems.
# It predicts categorical outputs by creating a tree-like
# structure of decisions.

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
# DataFrame organizes the data into rows and columns,
# making it compatible with Machine Learning algorithms.

# Step 4: Separate input (X) and output (y)
X = df[['Age', 'Income']]
y = df['Buy']

# Why use double brackets for X?
# X must be a 2-dimensional DataFrame because it contains
# multiple input features.

# Why use single brackets for y?
# y contains only one target column,
# so it is represented as a 1-dimensional Series.

# Step 5: Create Decision Tree model
model = DecisionTreeClassifier()

# Why create the model?
# This initializes the Decision Tree classifier before training.

# Step 6: Train the model
model.fit(X, y)

# What does fit() do?
# It trains the Decision Tree using the given dataset.
# The algorithm learns decision rules by splitting the data
# based on feature values.

# Step 7: Predict for a new customer
prediction = model.predict([[40, 50000]])

# Why use predict()?
# predict() uses the trained Decision Tree
# to classify new input data.

# Step 8: Display prediction
print("Prediction:", prediction[0])

# Why prediction[0]?
# predict() returns an array.
# prediction[0] extracts the first predicted value.

# ---------------------------------------------------
# SAMPLE OUTPUT
# ---------------------------------------------------

# Prediction: 1

# Interpretation:
# 1 → Customer Will Buy
# 0 → Customer Will Not Buy

# Note:
# The prediction may vary depending on the training data.