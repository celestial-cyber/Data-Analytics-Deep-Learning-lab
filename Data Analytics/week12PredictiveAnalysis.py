# THEORY:
# Predictive analytics uses historical data and machine learning
# algorithms to predict future values.
#
# In this program, Linear Regression is used to forecast
# future temperatures based on previous temperature records.
#
# Day number is the input feature, and temperature is the
# target variable.
#
# The model is evaluated using Mean Squared Error (MSE)
# and R-squared (R2) score.


# ALGORITHM:
# 1. Import the required libraries.
# 2. Create a sample weather dataset containing dates and temperatures.
# 3. Convert dates into day numbers.
# 4. Plot the historical temperature data.
# 5. Separate input features (X) and target values (y).
# 6. Split the dataset into training and testing sets.
# 7. Create and train a Linear Regression model.
# 8. Predict temperatures for the test data.
# 9. Calculate MSE and R-squared scores.
# 10. Predict temperatures for the next 7 days.
# 11. Plot historical and forecasted temperatures.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Create sample weather dataset
data = {
    'Day': pd.date_range('2022-01-01', periods=30),
    'Temperature': [25, 26, 27, 25, 28, 29, 30, 28, 27, 26,
                    25, 24, 23, 26, 27, 28, 29, 27, 25, 26,
                    28, 29, 30, 31, 32, 31, 29, 28, 27, 26]
}

df = pd.DataFrame(data)
df['Day_Num'] = df['Day'].dt.dayofyear

# Plot historical temperature
plt.plot(df['Day'], df['Temperature'], marker='o')
plt.xlabel('Day')
plt.ylabel('Temperature')
plt.title('Historical Temperature Data')
plt.grid()
plt.show()

# Define features and target
X = df[['Day_Num']]
y = df['Temperature']

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

# Forecast next 7 days
future_days = pd.date_range('2022-02-01', periods=7)
future_num = future_days.dayofyear.values.reshape(-1, 1)

future_temp = model.predict(future_num)

# Plot temperature forecast
plt.plot(df['Day'], df['Temperature'], 'bo-', label='Historical Temperature')
plt.plot(future_days, future_temp, 'ro--', label='Forecasted Temperature')

plt.xlabel('Day')
plt.ylabel('Temperature')
plt.title('Temperature Forecast')
plt.legend()
plt.grid()
plt.show()


# EXPECTED OUTPUT:
#
# 1. Mean Squared Error: A numerical value representing
#    the average squared prediction error.
#
# 2. R-squared: A numerical value representing how well
#    the model fits the test data.
#
# 3. First Graph:
#    Historical Temperature Data
#    Displays temperature variations over 30 days.
#
# 4. Second Graph:
#    Temperature Forecast
#    Blue line: Historical temperatures.
#    Red dotted line: Forecasted temperatures for 7 days.
#
# Note: Exact MSE, R-squared and forecast values are
# generated when the program runs.


# IMPORTANT KEY POINTS:
#
# 1. Predictive Analytics:
#    Uses historical data to predict future outcomes.
#
# 2. Linear Regression:
#    Predicts a continuous value using a linear relationship.
#
# 3. Feature (X):
#    Day_Num – Input variable.
#
# 4. Target (y):
#    Temperature – Value to be predicted.
#
# 5. train_test_split():
#    Divides data into training and testing sets.
#
# 6. model.fit():
#    Trains the Linear Regression model.
#
# 7. model.predict():
#    Predicts temperature values.
#
# 8. MSE:
#    Measures the average squared prediction error.
#    Lower MSE indicates better predictions.
#
# 9. R-squared:
#    Measures how well the model explains variations
#    in the target variable. A value closer to 1
#    generally indicates a better fit.
#
# 10. Matplotlib:
#     Used to visualize historical and forecasted data.
#
# 11. Pandas:
#     Used for dataset creation and date handling.
#
# 12. NumPy:
#     Used for numerical operations and reshaping data.
#
# NOTE:
# This is a simple demonstration using sample data.
# It uses day number alone as a feature, so it does not
# account for humidity, rainfall, wind speed or other
# factors that influence real-world weather.