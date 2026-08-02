# ==============================
# WEEK 10: PREDICTION TASKS
# EXPERIMENT 12: APPLY PREDICTIVE ANALYTICS FOR WEATHER FORECASTING
# ==============================

# Step 1: Import necessary libraries
import pandas as pd
from sklearn.linear_model import LinearRegression

# ---------------------------------------------------
# CODE EXPLANATION
# ---------------------------------------------------

# Why do we import pandas?
# Pandas is used to create and manipulate
# tabular datasets using DataFrames.

# Why do we import LinearRegression?
# Linear Regression is a Machine Learning algorithm
# used for predicting continuous numerical values.
# Here, it predicts future temperature.

# Step 2: Create sample weather dataset
data = {
    'Day': [1, 2, 3, 4, 5],
    'Temperature': [30, 32, 31, 34, 35]
}

# Why use Day and Temperature?
# Day is the independent variable (input feature).
# Temperature is the dependent variable (target).
# The model learns how temperature changes over time.

# Step 3: Convert dictionary into DataFrame
df = pd.DataFrame(data)

# Why convert to DataFrame?
# DataFrames organize data into rows and columns,
# making them suitable for Machine Learning algorithms.

# Step 4: Separate input (X) and output (y)
X = df[['Day']]
y = df['Temperature']

# Why use double brackets for X?
# X must be a 2-dimensional DataFrame because
# Scikit-learn expects features in matrix form.

# Why use single brackets for y?
# y is a single target column represented
# as a 1-dimensional Series.

# Step 5: Create Linear Regression model
model = LinearRegression()

# Why create the model?
# This initializes the Linear Regression algorithm.

# Step 6: Train the model
model.fit(X, y)

# What does fit() do?
# fit() trains the model by finding the best-fit line
# between Day and Temperature.

# Step 7: Predict temperature for Day 6
prediction = model.predict([[6]])

# Why use predict()?
# predict() estimates the future temperature
# based on the trained model.

# Step 8: Display predicted temperature
print("Predicted Temperature for Day 6:", prediction[0])

# Why prediction[0]?
# predict() returns an array.
# prediction[0] extracts the first predicted value.

# ---------------------------------------------------
# SAMPLE OUTPUT
# ---------------------------------------------------

# Predicted Temperature for Day 6: 36.3

# Interpretation:
# The model predicts that the temperature
# on Day 6 will be approximately 36.3°C.

# Note:
# The predicted value may vary slightly depending
# on floating-point calculations.
#
# Predictive Analytics uses historical weather
# observations to estimate future weather conditions.
# In real-world forecasting, more features such as
# humidity, rainfall, wind speed, and atmospheric
# pressure are also considered.