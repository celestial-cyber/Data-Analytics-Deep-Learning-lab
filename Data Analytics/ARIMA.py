# ==============================
# WEEK 6: ARIMA MODEL
# EXPERIMENT 7: IMPLEMENT ARIMA ON TIME SERIES DATA
# ==============================

# Step 1: Import necessary libraries
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# ---------------------------------------------------
# CODE EXPLANATION
# ---------------------------------------------------

# Why do we import pandas?
# Pandas is used to create and manipulate time-series data
# in the form of DataFrames and Series.

# Why do we import ARIMA?
# ARIMA (AutoRegressive Integrated Moving Average)
# is a forecasting algorithm used to predict future
# values based on past observations in a time series.

# Step 2: Create sample time series data
sales = [100, 120, 130, 125, 140, 150, 160, 170, 180, 190]

# Why are we using sales data?
# This represents monthly sales over time.
# ARIMA uses historical values to forecast future values.

# Step 3: Convert list into a Pandas Series
data = pd.Series(sales)

# Why convert to Series?
# ARIMA works with one-dimensional time series data.
# Pandas Series provides indexing and supports
# time-series operations.

# Step 4: Create ARIMA model
model = ARIMA(data, order=(1, 1, 1))

# What does order=(1,1,1) mean?
#
# order = (p, d, q)
#
# p = Auto Regression (AR)
# Number of previous observations used.
#
# d = Differencing
# Number of times data is differenced to make it stationary.
#
# q = Moving Average (MA)
# Number of previous forecast errors considered.

# Why use (1,1,1)?
# It is one of the simplest and most commonly used
# ARIMA configurations for demonstration purposes.

# Step 5: Train the ARIMA model
model_fit = model.fit()

# What does fit() do?
# fit() learns the patterns present in the historical
# time-series data and estimates ARIMA parameters.

# Step 6: Forecast the next 3 values
forecast = model_fit.forecast(steps=3)

# Why use forecast()?
# It predicts future observations using the trained model.

# Step 7: Display forecasted values
print("Forecasted Values:")
print(forecast)

# ---------------------------------------------------
# SAMPLE OUTPUT
# ---------------------------------------------------

# Forecasted Values:
#
# 10    199.8
# 11    209.3
# 12    218.6
#
# Note:
# The predicted values may vary slightly depending
# on the installed version of statsmodels and
# floating-point calculations.