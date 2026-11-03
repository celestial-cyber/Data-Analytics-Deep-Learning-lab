# THEORY:
# Predictive analytics uses historical sales data to predict
# future product sales.
#
# In this program, Linear Regression is used to forecast
# sales based on the month number.
#
# The model is evaluated using Mean Squared Error (MSE)
# and R-squared (R2) score.

# ALGORITHM:
# 1. Import the required libraries.
# 2. Create a sample product sales dataset.
# 3. Extract month numbers from the dates.
# 4. Plot historical sales data.
# 5. Define input features (X) and target variable (y).
# 6. Split the dataset into training and testing sets.
# 7. Create and train a Linear Regression model.
# 8. Predict sales on the test data.
# 9. Calculate MSE and R-squared scores.
# 10. Forecast sales for the next 6 months.
# 11. Plot historical and forecasted sales.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Create sample sales dataset
data = {
    'Month': pd.date_range('2022-01-01', periods=12, freq='ME'),
    'Sales': [1500, 1600, 1700, 1800, 1750, 1900,
              2100, 2200, 2300, 2400, 2500, 2600]
}

df = pd.DataFrame(data)
df['Month_Num'] = df['Month'].dt.month

# Plot historical sales
plt.plot(df['Month'], df['Sales'], 'bo-')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.title('Historical Sales Data')
plt.grid()
plt.show()

# Define features and target
X = df[['Month_Num']]
y = df['Sales']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)

print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R-squared:", r2_score(y_test, y_pred))

# Forecast future sales
future_months = pd.date_range('2023-01-01', periods=6, freq='ME')
future_num = future_months.month.values.reshape(-1, 1)

future_sales = model.predict(future_num)

# Plot sales forecast
plt.plot(df['Month'], df['Sales'], 'bo-', label='Historical Sales')
plt.plot(future_months, future_sales, 'ro--', label='Forecasted Sales')

plt.xlabel('Month')
plt.ylabel('Sales')
plt.title('Sales Forecast')
plt.legend()
plt.grid()
plt.show()


# EXPECTED OUTPUT:
#
# 1. Mean Squared Error:
#    Displays the average squared prediction error.
#
# 2. R-squared:
#    Displays how well the model fits the test data.
#
# 3. First Graph:
#    Historical Sales Data
#    Displays sales variations across 12 months.
#
# 4. Second Graph:
#    Sales Forecast
#    Blue line: Historical sales.
#    Red dotted line: Forecasted sales for 6 months.
#
# Note: Exact numerical values may vary depending on
# the model and dataset.

