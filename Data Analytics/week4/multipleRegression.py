# ============================================================
# MULTIPLE LINEAR REGRESSION
# House Price Prediction using Square Footage and Bedrooms
# ============================================================

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# train_test_split → splits the dataset into training and testing data
from sklearn.model_selection import train_test_split

# LinearRegression → creates the Multiple Linear Regression model
from sklearn.linear_model import LinearRegression

# Evaluation metrics:
# mean_squared_error → measures the average squared prediction error
# r2_score → measures how well the model explains the variation in the target
from sklearn.metrics import mean_squared_error, r2_score


# ============================================================
# PROGRAM DESCRIPTION
# ============================================================

# This program performs Multiple Linear Regression to predict
# the price of a house using two independent variables:
#
# 1. SquareFootage → size of the house
# 2. Bedrooms      → number of bedrooms
#
# Target variable:
# Price → price of the house
#
# Since we are using MORE THAN ONE independent variable,
# this is called MULTIPLE LINEAR REGRESSION.
#
# General equation:
#
# Price = b0 + b1(SquareFootage) + b2(Bedrooms)
#
# where:
# b0 = Intercept
# b1 = coefficient of SquareFootage
# b2 = coefficient of Bedrooms


# ============================================================
# SAMPLE DATASET
# ============================================================

# Creating a sample dataset using a Python dictionary
data = {
    'SquareFootage': [750, 900, 1100, 1250, 1400, 1600, 1800, 2000, 2300, 2600,
                       2800, 3000, 3200, 3500, 3800],

    'Bedrooms':      [1, 2, 2, 2, 3, 3, 3, 3, 4, 4,
                       4, 5, 5, 5, 6],

    'Price':         [150, 175, 205, 230, 255, 285, 310, 340, 385, 430,
                       460, 500, 535, 580, 625]
}

# pd.DataFrame() converts the dictionary into a table
df = pd.DataFrame(data)

# Display the dataset
print("Dataset:\n", df, "\n")


# ============================================================
# DEFINE FEATURES AND TARGET
# ============================================================

# X contains the independent/input variables (features)
#
# We have TWO features:
# 1. SquareFootage
# 2. Bedrooms
#
# Since there are multiple features, this is Multiple Regression.

X = df[['SquareFootage', 'Bedrooms']]

# y contains the dependent/target variable that we want to predict
y = df['Price']


# ============================================================
# SPLIT DATA INTO TRAINING AND TESTING SETS
# ============================================================

# train_test_split() divides the dataset into:
#
# Training data → used to train/learn the model
# Testing data  → used to check the model's performance
#
# test_size=0.2 means:
# 20% of the data is used for testing
# 80% of the data is used for training
#
# random_state=42 ensures that we get the same split
# every time we run the program.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ============================================================
# CREATE AND TRAIN THE LINEAR REGRESSION MODEL
# ============================================================

# LinearRegression() creates a Multiple Linear Regression model
model = LinearRegression()

# fit() trains the model using the training data
#
# The model learns the relationship between:
# SquareFootage + Bedrooms → Price

model.fit(X_train, y_train)


# ============================================================
# MAKE PREDICTIONS
# ============================================================

# predict() uses the trained model to predict house prices
# for the testing data.

y_pred = model.predict(X_test)


# ============================================================
# MODEL EVALUATION
# ============================================================

# We use two evaluation metrics:
#
# 1. Mean Squared Error (MSE)
# 2. R-squared (R²)


# ------------------------------------------------------------
# MEAN SQUARED ERROR (MSE)
# ------------------------------------------------------------

# mean_squared_error() calculates the average squared
# difference between the actual values and predicted values.
#
# Formula:
#
# MSE = (1/n) Σ (Actual - Predicted)²
#
# Smaller MSE → better model
#
# MSE = 0 means the predictions are exactly equal
# to the actual values.

mse = mean_squared_error(y_test, y_pred)


# ------------------------------------------------------------
# R-SQUARED (R²)
# ------------------------------------------------------------

# r2_score() calculates the R² score.
#
# R² tells us how well the independent variables
# explain the variation in the target variable.
#
# R² = 1    → perfect prediction
# R² = 0    → model does not explain the variation
# R² < 0    → model performs worse than a simple baseline
#
# Higher R² generally means a better fit.

r2 = r2_score(y_test, y_pred)


# ============================================================
# DISPLAY MODEL PARAMETERS
# ============================================================

print("Model Coefficients:")

# model.coef_[0] → coefficient of SquareFootage
# It tells how much the predicted price changes
# for one unit increase in SquareFootage,
# while keeping Bedrooms constant.

print(f"  SquareFootage coef: {model.coef_[0]:.4f}")

# model.coef_[1] → coefficient of Bedrooms
# It tells how much the predicted price changes
# for one additional bedroom,
# while keeping SquareFootage constant.

print(f"  Bedrooms coef:      {model.coef_[1]:.4f}")

# model.intercept_ → value of the target when
# all input features are zero.
#
# It represents b0 in the regression equation.

print(f"  Intercept:           {model.intercept_:.4f}\n")


# Display evaluation metrics
print("Mean Squared Error:", mse)
print("R-squared:", r2)


# ============================================================
# PREDICT PRICE FOR A NEW HOUSE
# ============================================================

# Create data for a new house:
# Square Footage = 2100
# Bedrooms = 3

new_house = pd.DataFrame({
    'SquareFootage': [2100],
    'Bedrooms': [3]
})

# predict() uses the trained model to predict
# the price of this new house.

predicted_price = model.predict(new_house)

print(f"\nPredicted price: {predicted_price[0]:.2f}K")


# ============================================================
# VISUALIZATION
# ============================================================

# np.linspace() creates 100 equally spaced values
# between the minimum and maximum square footage.
#
# These values are used to create a smooth regression line.

sqft_range = np.linspace(
    df['SquareFootage'].min(),
    df['SquareFootage'].max(),
    100
)

# Calculate the average number of bedrooms.
#
# Because this is MULTIPLE regression, there are two features.
# We cannot draw a single simple 2D regression line unless
# we keep one feature fixed.
#
# Here, Bedrooms is fixed at its average value.

avg_bedrooms = df['Bedrooms'].mean()


# Create input data for drawing the regression line
line_input = pd.DataFrame({
    'SquareFootage': sqft_range,
    'Bedrooms': avg_bedrooms
})

# Predict prices for these values
line_pred = model.predict(line_input)


# Create the plot
plt.figure(figsize=(7, 5))


# Scatter plot showing all actual house data
plt.scatter(
    df['SquareFootage'],
    df['Price'],
    color='blue',
    label='Actual Data (all houses)',
    s=60
)


# Show predictions made by the model for the test dataset
plt.scatter(
    X_test['SquareFootage'],
    y_pred,
    color='green',
    marker='X',
    s=100,
    label='Predicted (test set)'
)


# Plot the regression line.
#
# IMPORTANT:
# Since this is Multiple Linear Regression,
# this is a 2D representation of the regression surface.
#
# Bedrooms is kept fixed at its average value.

plt.plot(
    sqft_range,
    line_pred,
    color='red',
    linewidth=2,
    label='Regression Line (avg bedrooms)'
)


# Add labels to the graph
plt.xlabel('Square Footage')
plt.ylabel('Price ($1000s)')

# Add title
plt.title('House Price Prediction: Regression Line & Data')

# Display legend
plt.legend()

# Display grid
plt.grid(alpha=0.3)

# Show the final graph
plt.show()