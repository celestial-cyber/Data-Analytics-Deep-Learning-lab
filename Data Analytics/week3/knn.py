# ============================================================
# 2. IMPLEMENT ANY ONE IMPUTATION MODEL
# ============================================================


# ============================================================
# THEORY
# ============================================================

# In any real-world data collection, missing values can occur
# due to various reasons such as:
#
# 1. Errors in data entry
# 2. Non-response in surveys
# 3. Equipment malfunction
# 4. Data corruption
#
# Missing Value Imputation refers to the process of replacing
# missing data with suitable substituted or estimated values.
#
# Example:
#
# Before Imputation:
# Age = [20, 21, NaN, 23, 24]
#
# After Imputation:
# Age = [20, 21, 22, 23, 24]
#
# Here, NaN represents the missing value.


# ============================================================
# MISSING VALUE IMPUTATION METHODS
# ============================================================

# Some commonly used missing value imputation methods are:
#
# 1. Mean / Median / Mode Imputation
# 2. Predictive Imputation
# 3. Last Observation Carried Forward (LOCF)
#    and Next Observation Carried Backward (NOCB)


# ============================================================
# 1. MEAN / MEDIAN / MODE IMPUTATION
# ============================================================

# These are statistical methods used to replace missing values
# using statistical measures calculated from available values.
#
#
# MEAN IMPUTATION:
# Replaces missing values with the mean (average) of the
# available values.
#
# It is suitable for numerical data that does not contain
# significant outliers because outliers can affect the mean.
#
#
# MEDIAN IMPUTATION:
# Replaces missing values with the median of the available
# values.
#
# It is more robust than mean imputation when the data contains
# outliers or is not normally distributed.
#
#
# MODE IMPUTATION:
# Replaces missing values with the mode, which is the most
# frequently occurring value.
#
# It is commonly used for categorical data.


# ============================================================
# 2. PREDICTIVE IMPUTATION
# ============================================================

# Predictive imputation uses statistical or machine-learning
# models to predict and fill missing values based on relationships
# present in the remaining data.
#
# Common predictive imputation methods include:
#
# 1. Regression Imputation
#    → Uses a regression model to predict missing values
#      using other related variables.
#
# 2. K-Nearest Neighbors (KNN) Imputation
#    → Identifies 'k' samples that are similar to the observation
#      containing the missing value.
#    → The values of these nearest neighbours are used to
#      estimate the missing value.


# ============================================================
# K-NEAREST NEIGHBORS (KNN) IMPUTATION
# ============================================================

# In this program, we implement PREDICTIVE IMPUTATION
# using K-Nearest Neighbors (KNN).
#
# KNN stands for K-Nearest Neighbors.
#
# KNN Imputation identifies the 'k' most similar observations
# (nearest neighbours) and uses their available values to
# estimate the missing value.
#
# For numerical data, the missing value is generally estimated
# using the average of the values from the selected neighbours.
#
# In this program:
#
# K = 2
#
# Therefore, 2 nearest neighbours are considered to estimate
# the missing value.
#
# We use KNNImputer from the Scikit-learn library.


# ============================================================
# PROGRAM
# ============================================================

import numpy as np
import pandas as pd

# KNNImputer is used to perform KNN-based missing value
# imputation.
from sklearn.impute import KNNImputer


# Create a dataset containing a missing value.
#
# np.nan represents a missing value.

data = {
    'Feature1': [25, 20, 30, 40, 29, 27, 32, 31],
    'Feature2': [20, 25, np.nan, 45, 30, 25, 35, 40]
}


# Convert the dictionary into a Pandas DataFrame.

df = pd.DataFrame(data)


# Display the original dataset.

print("Original Dataset:")
print(df)


# ============================================================
# PREDICTIVE IMPUTATION USING KNN
# ============================================================

# Create the KNN Imputation model.
#
# n_neighbors=2 means that 2 nearest/similar observations
# will be considered while estimating the missing value.

imputer = KNNImputer(n_neighbors=2)


# fit_transform() performs two operations:
#
# fit()       → learns the required information from the data
# transform() → replaces the missing value
#
# The resulting dataset is stored in df_filled.

df_filled = imputer.fit_transform(df)


# Display the dataset after imputation.

print("\nDataset After KNN Imputation:")
print(df_filled)


# ============================================================
# IMPORTANT NOTES
# ============================================================

# np.nan
# → Represents a missing value in NumPy.
#
#
# pd.DataFrame()
# → Converts the given data into a tabular DataFrame.
#
#
# KNNImputer()
# → Creates a KNN-based missing value imputation model.
#
#
# n_neighbors=2
# → Specifies that 2 nearest neighbours should be considered.
#
#
# fit_transform()
# → Learns from the data and replaces the missing values.
#
#
# df_filled
# → Contains the dataset after missing values have been
#   replaced with estimated values.


# ============================================================
# EXPECTED OUTPUT
# ============================================================

# Original Dataset:
#
#    Feature1  Feature2
# 0        25      20.0
# 1        20      25.0
# 2        30       NaN
# 3        40      45.0
# 4        29      30.0
# 5        27      25.0
# 6        32      35.0
# 7        31      40.0
#
#
# Dataset After KNN Imputation:
#
# [[25.  20. ]
#  [20.  25. ]
#  [30.  27.5]
#  [40.  45. ]
#  [29.  30. ]
#  [27.  25. ]
#  [32.  35. ]
#  [31.  40. ]]
#
#
# The missing value NaN in Feature2 is replaced by 27.5.


# ============================================================
# HOW THE MISSING VALUE IS ESTIMATED
# ============================================================

# The missing value occurs when:
#
# Feature1 = 30
# Feature2 = NaN
#
# KNNImputer identifies the nearest/similar observations
# and uses their available Feature2 values.
#
# The selected values are:
#
# 25 and 30
#
# Average:
#
# (25 + 30) / 2 = 27.5
#
# Therefore:
#
# NaN → 27.5


# ============================================================
# CONCLUSION
# ============================================================

# KNN Imputation successfully replaces the missing value by
# using information from the nearest similar observations.
#
# KNN is a predictive imputation technique because it uses
# relationships between observations to estimate missing data.