# ==============================
# WEEK 9: ANALYTICS TECHNIQUES
# EXPERIMENT 11: PERFORM PREDICTIVE ANALYTICS ON PRODUCT SALES DATA
# ==============================

# Step 1: Import necessary libraries
import pandas as pd
from sklearn.linear_model import LinearRegression

# ---------------------------------------------------
# CODE EXPLANATION
# ---------------------------------------------------

# Why do we import pandas?
# Pandas is used to create and manipulate tabular
# datasets using DataFrames.

# Why do we import LinearRegression?
# Linear Regression is a Machine Learning algorithm
# used for predictive analytics.
# It predicts future numerical values based on
# historical data.

# Step 2: Create sample product sales dataset
data = {
    'Month': [1, 2, 3, 4, 5],
    'Sales': [1200, 1500, 1800, 2100, 2400]
}

# Why use Month and Sales?
# Month is the independent variable (input).
# Sales is the dependent variable (output).
# The model learns the relationship between
# month and sales.

# Step 3: Convert dictionary into DataFrame
df = pd.DataFrame(data)

# Why convert to DataFrame?
# DataFrames organize data into rows and columns
# and are compatible with Machine Learning libraries.

# Step 4: Separate input (X) and output (y)
X = df[['Month']]
y = df['Sales']

# Why use double brackets for X?
# X must be a 2-dimensional DataFrame.

# Why use single brackets for y?
# y is the target variable represented as
# a 1-dimensional Series.

# Step 5: Create Linear Regression model
model = LinearRegression()

# Why create the model?
# It initializes the Linear Regression algorithm.

# Step 6: Train the model
model.fit(X, y)

# What does fit() do?
# fit() trains the model by learning the relationship
# between Month and Sales.

# Step 7: Predict sales for Month 6
prediction = model.predict([[6]])

# Why use predict()?
# predict() estimates future sales using
# the trained model.

# Step 8: Display predicted sales
print("Predicted Sales for Month 6:", prediction[0])

# Why prediction[0]?
# predict() returns an array.
# prediction[0] extracts the predicted value.

# ---------------------------------------------------
# SAMPLE OUTPUT
# ---------------------------------------------------

# Predicted Sales for Month 6: 2700.0

# Interpretation:
# The model predicts that product sales for
# Month 6 will be approximately 2700 units.

# Note:
# The prediction is based on the linear trend
# learned from the historical sales data.
#
# Predictive Analytics uses historical data
# to forecast future outcomes and support
# business decision-making.