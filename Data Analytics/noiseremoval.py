# ==============================
# NOISE DETECTION AND REMOVAL USING IQR
# ==============================

# CONCEPT OVERVIEW
# This program detects and removes noisy values (outliers) from a dataset.
# It uses the Interquartile Range (IQR) method to identify values that are
# significantly different from the rest of the data.

# WHY IS THIS CONCEPT USED?
# Real-world datasets often contain incorrect or abnormal values due to
# sensor errors, manual data entry mistakes, transmission errors, or
# equipment faults. These values can affect data analysis and machine
# learning models. Removing them improves accuracy.

# REAL-WORLD APPLICATIONS
# • Machine Learning data preprocessing
# • Medical data analysis
# • Banking fraud detection
# • Sensor data cleaning
# • Sales and business analytics
# • Stock market analysis

# THEORY
# Noise refers to unwanted or abnormal values present in a dataset.
# Such values are called outliers because they are far away from the
# majority of the data.

# WHAT IS IQR?
# IQR (Interquartile Range) measures the spread of the middle 50%
# of the data.
#
# Formula:
# IQR = Q3 - Q1
#
# Q1 = 25th Percentile
# Q3 = 75th Percentile

# LOWER AND UPPER BOUNDS
# Lower Bound = Q1 - (1.5 × IQR)
# Upper Bound = Q3 + (1.5 × IQR)
#
# Any value outside these bounds is considered an outlier and is removed.

# LINE-BY-LINE EXPLANATION

# Step 1:
# Import the NumPy library.
# NumPy provides functions for numerical calculations and percentile computation.

# Step 2:
# Create a sample dataset containing normal values and two obvious outliers
# (500 and 999).

# Step 3:
# Convert the Python list into a NumPy array so mathematical operations
# can be performed efficiently.

# Step 4:
# Calculate:
# Q1 = First Quartile (25th percentile)
# Q3 = Third Quartile (75th percentile)
# IQR = Difference between Q3 and Q1

# Step 5:
# Compute the lower and upper limits.
# Any value outside these limits is treated as noise.

# Step 6:
# Apply Boolean indexing to keep only those values that lie within
# the lower and upper bounds.
# This removes the noisy values.

# Step 7:
# Print both the original dataset and the cleaned dataset.

# DRY RUN

# Original Data:
# [10, 12, 11, 13, 12, 500, 13, 14, 12, 999, 11, 10]

# Q1 = 11
# Q3 = 13
# IQR = 2

# Lower Bound = 11 - (1.5 × 2) = 8
# Upper Bound = 13 + (1.5 × 2) = 16

# Values between 8 and 16 are retained.

# Removed Values:
# 500
# 999

# Final Cleaned Data:
# [10 12 11 13 12 13 14 12 11 10]

# MEMORY VISUALIZATION

# data
# ┌───────────────────────────────┐
# │10 12 11 13 12 500 13 14 12... │
# └───────────────────────────────┘

# clean_data
# ┌───────────────────────────────┐
# │10 12 11 13 12 13 14 12 11 10 │
# └───────────────────────────────┘

# OUTPUT EXPLANATION
# The values 500 and 999 lie outside the calculated IQR limits.
# Therefore, they are identified as outliers and removed.
# Only valid data values remain in the cleaned dataset.

# TIME COMPLEXITY
# Best Case: O(n)
# Average Case: O(n)
# Worst Case: O(n)
#
# Reason:
# The dataset is scanned once to filter the values.

# SPACE COMPLEXITY
# O(n)
#
# Because a new array (clean_data) is created.

# ALTERNATIVE APPROACHES

# 1. Z-Score Method
# Measures how many standard deviations a value is from the mean.

# 2. Modified Z-Score
# Uses the median instead of the mean.

# 3. Isolation Forest
# Machine learning algorithm for anomaly detection.

# 4. DBSCAN
# Detects outliers based on density.

# COMPARISON

# IQR
# • Simple
# • Robust
# • Best for small datasets

# Z-Score
# • Assumes normal distribution
# • Sensitive to extreme values

# Isolation Forest
# • Suitable for large datasets
# • More computationally expensive

# COMMON MISTAKES

# • Forgetting to convert the list into a NumPy array.
# • Using incorrect percentile values.
# • Using AND (&) incorrectly in Boolean indexing.
# • Confusing IQR with range.
# • Forgetting parentheses around conditions.

# REAL-WORLD ANALOGY

# Imagine a classroom where every student's height is between
# 150 cm and 180 cm. If one recorded height is 400 cm,
# it is clearly an incorrect measurement.
# IQR identifies such unrealistic values and removes them.

# VIVA QUESTIONS

# Q1. What is noise?
# Noise is unwanted or abnormal data present in a dataset.

# Q2. What is an outlier?
# An outlier is a value significantly different from the rest of the data.

# Q3. What is IQR?
# IQR is the difference between the third quartile (Q3) and first quartile (Q1).

# Q4. Why do we use IQR?
# To detect and remove outliers without assuming normal distribution.

# Q5. What are Q1 and Q3?
# Q1 is the 25th percentile and Q3 is the 75th percentile.

# Q6. Why multiply IQR by 1.5?
# It is the standard statistical rule for identifying outliers.

# Q7. Why is NumPy used?
# NumPy provides efficient numerical operations and percentile calculations.

# EXAM TIPS

# • Define noise and outliers.
# • Write the IQR formula.
# • Mention lower and upper bound formulas.
# • Explain why 500 and 999 are removed.
# • State the applications of IQR.

# SUMMARY

# • Noise = unwanted values.
# • IQR = Q3 - Q1.
# • Lower Bound = Q1 - 1.5 × IQR.
# • Upper Bound = Q3 + 1.5 × IQR.
# • Values outside these limits are removed.
# • IQR is a simple and robust outlier detection method.
# Step 1: Import necessary library
import numpy as np

# Step 2: Example data with some noisy (outlier) values
data = [10, 12, 11, 13, 12, 500, 13, 14, 12, 999, 11, 10]

# Step 3: Convert to NumPy array
data = np.array(data)

# Step 4: Calculate IQR
Q1 = np.percentile(data, 25)
Q3 = np.percentile(data, 75)
IQR = Q3 - Q1

# Step 5: Define lower and upper bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Step 6: Remove noisy data (outliers)
clean_data = data[(data >= lower_bound) & (data <= upper_bound)]

# Step 7: Print results
print("Original Data:", data)
print("Cleaned Data:", clean_data)