# Data preprocessing is the process of cleaning and preparing data for analysis.
# Missing values occur when data is not available in a dataset.
# They can be handled using different methods:
# 1. dropna() - Removes rows containing missing values.
# 2. fillna(0) - Replaces missing values with 0.
# 3. Mean - Replaces missing numerical values with the column average.
# 4. Forward Fill - Uses the previous available value.
# 5. Backward Fill - Uses the next available value.
# 6. Interpolation - Estimates missing numerical values using nearby values.

# Step 1: Import pandas and numpy libraries.
# Step 2: Create a DataFrame containing missing values.
# Step 3: Display the original DataFrame.
# Step 4: Use dropna() to remove rows with missing values.
# Step 5: Use fillna(0) to replace missing values with zero.
# Step 6: Replace missing numerical values with the column mean.
# Step 7: Apply forward fill and backward fill.
# Step 8: Apply interpolation to numerical columns.
# Step 9: Display the results of all methods.

import pandas as pd
import numpy as np

# Create sample data
data = {
    'A': [1, 2, np.nan, 4, 5],
    'B': [np.nan, 2, 3, np.nan, 5],
    'C': ['foo', 'bar', 'baz', np.nan, 'qux']
}

df = pd.DataFrame(data)

print("Original DataFrame:")
print(df)

# Method 1: Remove missing values
df_dropna = df.dropna()
print("\nAfter removing missing values:")
print(df_dropna)

# Method 2: Fill missing values with 0
df_fillna = df.fillna(0)
print("\nAfter filling with 0:")
print(df_fillna)

# Method 3: Fill numerical columns with mean
df_mean = df.copy()
df_mean['A'] = df_mean['A'].fillna(df_mean['A'].mean())
df_mean['B'] = df_mean['B'].fillna(df_mean['B'].mean())

print("\nAfter mean imputation:")
print(df_mean)

# Method 4: Forward fill
df_ffill = df.ffill()
print("\nAfter forward fill:")
print(df_ffill)

# Method 5: Backward fill
df_bfill = df.bfill()
print("\nAfter backward fill:")
print(df_bfill)

# Method 6: Interpolation
df_interp = df.copy()
df_interp[['A', 'B']] = df_interp[['A', 'B']].interpolate()

print("\nAfter interpolation:")
print(df_interp)


# Original DataFrame:
#      A    B    C
# 0  1.0  NaN  foo
# 1  2.0  2.0  bar
# 2  NaN  3.0  baz
# 3  4.0  NaN  NaN
# 4  5.0  5.0  qux
#
# After removing missing values:
#      A    B    C
# 1  2.0  2.0  bar
# 4  5.0  5.0  qux
#
# After filling with 0:
#      A    B    C
# 0  1.0  0.0  foo
# 1  2.0  2.0  bar
# 2  0.0  3.0  baz
# 3  4.0  0.0    0
# 4  5.0  5.0  qux
#
# After mean imputation:
#      A    B    C
# 0  1.0  3.333333 foo
# 1  2.0  2.000000 bar
# 2  3.0  3.000000 baz
# 3  4.0  3.333333 NaN
# 4  5.0  5.000000 qux
#
# After forward fill:
#      A    B    C
# 0  1.0  NaN  foo
# 1  2.0  2.0  bar
# 2  2.0  3.0  baz
# 3  4.0  3.0  baz
# 4  5.0  5.0  qux
#
# After backward fill:
#      A    B    C
# 0  1.0  2.0  foo
# 1  2.0  2.0  bar
# 2  4.0  3.0  baz
# 3  4.0  5.0  qux
# 4  5.0  5.0  qux
#
# After interpolation:
#      A    B    C
# 0  1.0  NaN  foo
# 1  2.0  2.0  bar
# 2  3.0  3.0  baz
# 3  4.0  4.0  NaN
# 4  5.0  5.0  qux

# 1. Missing values are represented by NaN.
# 2. dropna() removes rows containing missing values.
# 3. fillna() replaces missing values.
# 4. Mean imputation is used for numerical columns.
# 5. Forward fill uses the previous available value.
# 6. Backward fill uses the next available value.
# 7. Interpolation estimates missing numerical values.
# 8. copy() creates a separate copy of the DataFrame.
# 9. Mean imputation should not be applied directly to categorical columns.


# Q1. What is data preprocessing?
# Ans: It is the process of cleaning and preparing data for analysis.

# Q2. What are missing values?
# Ans: Values that are not available in a dataset.

# Q3. How are missing values represented in pandas?
# Ans: They are commonly represented as NaN.

# Q4. What is dropna()?
# Ans: It removes rows or columns containing missing values.

# Q5. What is fillna()?
# Ans: It replaces missing values with a specified value.

# Q6. What is mean imputation?
# Ans: It replaces missing numerical values with the column mean.

# Q7. What is forward fill?
# Ans: It fills missing values using the previous available value.

# Q8. What is backward fill?
# Ans: It fills missing values using the next available value.

# Q9. What is interpolation?
# Ans: It estimates missing numerical values using nearby values.

# Q10. What is the difference between dropna() and fillna()?
# Ans: dropna() removes missing-value rows, while fillna() replaces missing values.

# Q11. Why is data preprocessing important?
# Ans: It improves data quality and prepares data for analysis.

# Q12. Can mean imputation be used for categorical data?
# Ans: No. Most frequent value imputation is generally used for categorical data.

