# ==============================
# WEEK 4: REGRESSION CONCEPTS
# EXPERIMENT 4: IMPLEMENT LOGISTIC REGRESSION
# ==============================

# Step 1: Import necessary libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression

# ---------------------------------------------------
# CODE EXPLANATION
# ---------------------------------------------------

# Why do we import pandas?
# Pandas is used to create and manipulate tabular data
# using DataFrames.

# Why do we import LogisticRegression?
# LogisticRegression is a Machine Learning algorithm
# used for classification problems.
# It predicts categories such as Yes/No, Pass/Fail,
# Spam/Not Spam, etc.

# Step 2: Create sample dataset
data = {
    'Hours': [1, 2, 3, 4, 5, 6, 7, 8],
    'Pass':  [0, 0, 0, 0, 1, 1, 1, 1]
}

# Why are we using Hours and Pass?
# Hours is the independent variable (input feature).
# Pass is the dependent variable (target).
# 0 means Fail.
# 1 means Pass.

# Step 3: Convert dictionary into DataFrame
df = pd.DataFrame(data)

# Why convert to DataFrame?
# Machine Learning models work efficiently with
# tabular data organized into rows and columns.

# Step 4: Separate input (X) and output (y)
X = df[['Hours']]
y = df['Pass']

# Why use double brackets for X?
# df[['Hours']] returns a DataFrame (2D),
# which is required by Scikit-learn.

# Why use single brackets for y?
# df['Pass'] returns a Series (1D),
# which represents the target variable.

# Step 5: Create Logistic Regression model
model = LogisticRegression()

# Why create the model?
# This initializes the Logistic Regression algorithm
# before training.

# Step 6: Train the model
model.fit(X, y)

# What does fit() do?
# fit() trains the model using the training dataset.
# It learns the relationship between study hours
# and the probability of passing.

# Step 7: Predict result for a student who studied 5 hours
prediction = model.predict([[5]])

# Why use predict()?
# predict() uses the trained model to classify
# new input data.

# Step 8: Display prediction
print("Prediction for 5 Hours:", prediction[0])

# Why prediction[0]?
# predict() returns an array.
# prediction[0] extracts the first prediction.

# ---------------------------------------------------
# SAMPLE OUTPUT
# ---------------------------------------------------

# Prediction for 5 Hours: 1

# Interpretation:
# 1 → Pass
# 0 → Fail
#
# The model predicts that a student studying
# for 5 hours is likely to pass.
