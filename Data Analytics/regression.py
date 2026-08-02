# ==============================
# WEEK 4: REGRESSION CONCEPTS
# EXPERIMENT 3: IMPLEMENT LINEAR REGRESSION
# ==============================

# Step 1: Import necessary libraries
import pandas as pd
from sklearn.linear_model import LinearRegression

# ---------------------------------------------------
# CODE EXPLANATION
# ---------------------------------------------------

# Why do we import pandas?
# Pandas is used to create and manipulate tabular data
# in the form of DataFrames.

# Why do we import LinearRegression?
# LinearRegression is a Machine Learning algorithm
# provided by Scikit-learn to predict continuous values
# based on a linear relationship between variables.

# Step 2: Create sample dataset
data = {
    'Hours': [1, 2, 3, 4, 5],
    'Marks': [35, 50, 65, 80, 95]
}

# Why are we using Hours and Marks?
# Hours is the independent variable (input).
# Marks is the dependent variable (output).
# We want to predict marks based on study hours.

# Step 3: Convert dictionary into DataFrame
df = pd.DataFrame(data)

# Why convert to DataFrame?
# DataFrames organize data into rows and columns
# and are compatible with Machine Learning libraries.

# Step 4: Separate input (X) and output (y)
X = df[['Hours']]
y = df['Marks']

# Why use X and y?
# X contains the independent variable (feature).
# y contains the dependent variable (target).

# Why double brackets for X?
# df[['Hours']] returns a DataFrame (2D array),
# which is required by Scikit-learn.

# Why single brackets for y?
# df['Marks'] returns a Series (1D array),
# which is required as the target variable.

# Step 5: Create Linear Regression model
model = LinearRegression()

# Why create the model?
# This initializes the Linear Regression algorithm
# before training.

# Step 6: Train the model
model.fit(X, y)

# What does fit() do?
# fit() trains the model using the given data.
# It finds the best-fit straight line by calculating
# the slope and intercept.

# Step 7: Predict marks for a student who studied 6 hours
prediction = model.predict([[6]])

# Why use predict()?
# predict() uses the trained model to estimate
# the output for new input values.

# Step 8: Display the prediction
print("Predicted Marks for 6 Hours:", prediction[0])

# Why prediction[0]?
# predict() returns an array.
# prediction[0] extracts the first predicted value.

# ---------------------------------------------------
# SAMPLE OUTPUT
# ---------------------------------------------------

# Predicted Marks for 6 Hours: 110.0

# Note:
# The exact output may vary slightly depending on
# floating-point calculations.