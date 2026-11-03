# THEORY:
# Time Series Analysis is used to analyze data collected over time.
#
# ARIMA stands for AutoRegressive Integrated Moving Average.
#
# It is a statistical model used to analyze past observations
# and forecast future values.
#
# ARIMA consists of three parameters:
# 1. AR (p): Uses previous values for prediction.
# 2. I (d): Makes the data stationary using differencing.
# 3. MA (q): Uses previous forecast errors for prediction.
#
# In this program, ARIMA(2,1,2) is used to forecast
# the next 12 months of time series data.


import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Create sample time series data
data = {
    'Date': pd.date_range('2022-01-01', periods=24, freq='ME'),
    'Value': [120, 130, 135, 140, 150, 160, 170, 175,
              180, 190, 200, 210, 220, 230, 240, 250,
              260, 270, 280, 290, 300, 310, 320, 330]
}

df = pd.DataFrame(data).set_index('Date')

# Plot original time series
df['Value'].plot(title='Time Series Data')
plt.xlabel('Date')
plt.ylabel('Value')
plt.show()

# Plot ACF and PACF
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
plot_acf(df['Value'], lags=20, ax=ax[0])
plot_pacf(df['Value'], lags=20, ax=ax[1])
plt.show()

# Fit ARIMA model
model = ARIMA(df['Value'], order=(2, 1, 2))
fit = model.fit()

# Display model summary
print(fit.summary())

# Forecast next 12 months
forecast = fit.forecast(steps=12)

dates = pd.date_range(
    start=df.index[-1] + pd.DateOffset(months=1),
    periods=12, freq='ME'
)

forecast_df = pd.DataFrame(forecast.values, index=dates, columns=['Forecast'])

# Plot forecast
plt.plot(df['Value'], label='Original Data')
plt.plot(forecast_df['Forecast'], 'r--', label='Forecast')
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('ARIMA Forecast')
plt.legend()
plt.show()


# EXPECTED OUTPUT:
#
# 1. Time Series Graph:
#    Displays the original values over 24 months.
#
# 2. ACF Graph:
#    Shows the correlation between observations and
#    their previous values at different lags.
#
# 3. PACF Graph:
#    Shows the direct correlation between observations
#    and their lagged values.
#
# 4. Model Summary:
#    Displays ARIMA model parameters and statistical results.
#
# 5. Forecast Graph:
#    Blue line: Original time series data.
#    Red dotted line: Forecasted values for the next 12 months.


# IMPORTANT KEY POINTS:
#
# 1. Time Series:
#    Data collected at regular time intervals.
#
# 2. ARIMA:
#    AutoRegressive Integrated Moving Average.
#
# 3. AR (p):
#    Uses previous observations for prediction.
#
# 4. I (d):
#    Uses differencing to make data stationary.
#
# 5. MA (q):
#    Uses past forecast errors.
#
# 6. Stationarity:
#    A time series whose statistical properties remain
#    approximately constant over time.
#
# 7. ACF:
#    Autocorrelation Function measures correlation
#    between observations at different lags.
#
# 8. PACF:
#    Partial Autocorrelation Function measures direct
#    correlation between observations at different lags.
#
# 9. ARIMA(order=(2,1,2)):
#    p = 2, d = 1, q = 2.
#
# 10. model.fit():
#     Trains the ARIMA model using the given data.
#
# 11. forecast(steps=12):
#     Predicts the next 12 time periods.
#
# 12. Statsmodels:
#     Python library used for statistical time series analysis.
#
# 13. Matplotlib:
#     Used to visualize original and forecasted data.

# VIVA QUESTIONS:
#
# 1. What is time series analysis?
#    It is the analysis of data collected over time.
#
# 2. What is ARIMA?
#    ARIMA stands for AutoRegressive Integrated Moving Average.
#    It is used for time series forecasting.
#
# 3. What are the three components of ARIMA?
#    AR (p), I (d) and MA (q).
#
# 4. What is the meaning of AR?
#    AutoRegression uses previous observations to predict
#    future values.
#
# 5. What is the meaning of I?
#    Integration uses differencing to make the time series
#    stationary.
#
# 6. What is the meaning of MA?
#    Moving Average uses previous forecast errors.
#
# 7. What is stationarity?
#    A time series is stationary when its statistical
#    properties remain approximately constant over time.
#
# 8. What is ACF?
#    ACF measures the correlation between observations
#    and their lagged values.
#
# 9. What is PACF?
#    PACF measures the direct correlation between
#    observations at different lags.
#
# 10. What does ARIMA(2,1,2) mean?
#     p = 2, d = 1 and q = 2.
#
# 11. Which library is used for ARIMA?
#     Statsmodels.
#
# 12. What is forecasting?
#     Forecasting is predicting future values using
#     historical data.
#
# 13. What does forecast(steps=12) do?
#     It predicts the next 12 time periods.
#
# 14. What is the use of differencing?
#     It helps remove trends and make a time series stationary.

