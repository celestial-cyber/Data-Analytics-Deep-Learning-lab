# ==============================
# WEEK 9: ANALYTICS TECHNIQUES
# EXPERIMENT 10: PERFORM DESCRIPTIVE ANALYTICS ON HEALTHCARE DATA
# ==============================

# Step 1: Import necessary library
import pandas as pd

# ---------------------------------------------------
# CODE EXPLANATION
# ---------------------------------------------------

# Why do we import pandas?
# Pandas is used to create, organize, analyze,
# and manipulate tabular datasets using DataFrames.

# Step 2: Create sample healthcare dataset
data = {
    'Patient': ['P1', 'P2', 'P3', 'P4', 'P5'],
    'Age': [25, 40, 35, 50, 30],
    'BloodPressure': [120, 140, 130, 150, 125],
    'HeartRate': [72, 80, 76, 85, 74]
}

# Why use this dataset?
# It contains basic healthcare information such as
# patient age, blood pressure, and heart rate.
# Descriptive Analytics summarizes this data
# using statistical measures.

# Step 3: Convert dictionary into DataFrame
df = pd.DataFrame(data)

# Why convert to DataFrame?
# DataFrames organize data into rows and columns
# making it easier to analyze.

# Step 4: Display the original dataset
print("Healthcare Dataset:")
print(df)

# Why print the dataset?
# To view the original healthcare records
# before performing analytics.

# Step 5: Generate descriptive statistics
summary = df.describe()

# What does describe() do?
# It calculates statistical measures such as:
# Count
# Mean
# Standard Deviation
# Minimum
# 25% Quartile
# Median (50%)
# 75% Quartile
# Maximum

# Step 6: Display descriptive statistics
print("\nDescriptive Analytics:")
print(summary)

# Why print summary?
# To understand the overall characteristics
# of the healthcare dataset.

# ---------------------------------------------------
# SAMPLE OUTPUT
# ---------------------------------------------------

# Healthcare Dataset:
#
#   Patient  Age  BloodPressure  HeartRate
# 0      P1   25             120         72
# 1      P2   40             140         80
# 2      P3   35             130         76
# 3      P4   50             150         85
# 4      P5   30             125         74
#
#
# Descriptive Analytics:
#
#              Age  BloodPressure  HeartRate
# count   5.000000        5.000000   5.000000
# mean   36.000000      133.000000  77.400000
# std     9.617692       12.041595   5.458938
# min    25.000000      120.000000  72.000000
# 25%    30.000000      125.000000  74.000000
# 50%    35.000000      130.000000  76.000000
# 75%    40.000000      140.000000  80.000000
# max    50.000000      150.000000  85.000000

# Note:
# The describe() function automatically computes
# descriptive statistics for all numerical columns.
#
# These statistics help summarize and understand
# the healthcare dataset before further analysis.