# Data redundancy occurs when duplicate or unnecessary data is present.
# It increases storage requirements and may affect data analysis.
# Redundancy can be removed using the following methods:
# 1. drop_duplicates() - Removes duplicate rows.
# 2. Duplicate column removal - Removes columns with duplicate names.
# 3. Correlation analysis - Identifies highly correlated columns.
# 4. drop() - Removes unnecessary columns.

# Step 1: Import pandas, numpy and required libraries.
# Step 2: Create a DataFrame containing redundant data.
# Step 3: Display the original DataFrame.
# Step 4: Remove duplicate rows using drop_duplicates().
# Step 5: Remove duplicate column names.
# Step 6: Calculate the correlation matrix.
# Step 7: Identify columns having correlation greater than 0.9.
# Step 8: Remove highly correlated columns.
# Step 9: Display the final DataFrame.

import pandas as pd
import numpy as np

# Create sample data
data = {
    'A': [1, 2, 2, 4, 5],
    'B': [10, 20, 20, 40, 50],
    'C': [1, 2, 2, 4, 5]
}

df = pd.DataFrame(data)

print("Original DataFrame:")
print(df)

# Remove duplicate rows
df_no_duplicates = df.drop_duplicates()

print("\nAfter removing duplicate rows:")
print(df_no_duplicates)

# Remove duplicate column names
df_no_duplicates = df_no_duplicates.loc[
    :, ~df_no_duplicates.columns.duplicated()
]

print("\nAfter removing duplicate columns:")
print(df_no_duplicates)

# Calculate correlation matrix
correlation_matrix = df_no_duplicates.corr().abs()

# Identify highly correlated columns
upper = correlation_matrix.where(
    np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
)

to_drop = [
    column for column in upper.columns
    if any(upper[column] > 0.9)
]

# Remove highly correlated columns
df_final = df_no_duplicates.drop(columns=to_drop)

print("\nAfter removing highly correlated columns:")
print(df_final)

# Original DataFrame:
#    A   B  C
# 0  1  10  1
# 1  2  20  2
# 2  2  20  2
# 3  4  40  4
# 4  5  50  5
#
# After removing duplicate rows:
#    A   B  C
# 0  1  10  1
# 1  2  20  2
# 3  4  40  4
# 4  5  50  5
#
# After removing duplicate columns:
#    A   B  C
# 0  1  10  1
# 1  2  20  2
# 3  4  40  4
# 4  5  50  5
#
# After removing highly correlated columns:
#    A
# 0  1
# 1  2
# 3  4
# 4  5

# 1. Data redundancy means unnecessary repetition of data.
# 2. drop_duplicates() removes duplicate rows.
# 3. columns.duplicated() identifies duplicate column names.
# 4. corr() calculates the correlation between numerical columns.
# 5. Correlation values range from -1 to +1.
# 6. abs() converts negative correlation values into positive values.
# 7. Correlation greater than 0.9 indicates a strong positive relationship.
# 8. drop() removes selected columns.
# 9. Redundancy removal reduces unnecessary data.

# Q1. What is data redundancy?
# Ans: It is the unnecessary repetition of data.

# Q2. What is data cleaning?
# Ans: It is the process of identifying and correcting unwanted data.

# Q3. Which function removes duplicate rows?
# Ans: drop_duplicates().

# Q4. What is correlation?
# Ans: It measures the relationship between two numerical variables.

# Q5. What is a correlation matrix?
# Ans: It shows the correlation values between numerical columns.

# Q6. What is the range of correlation?
# Ans: -1 to +1.

# Q7. What does correlation = +1 mean?
# Ans: Perfect positive correlation.

# Q8. What does correlation = -1 mean?
# Ans: Perfect negative correlation.

# Q9. What does correlation = 0 mean?
# Ans: No linear correlation.

# Q10. What is the purpose of drop()?
# Ans: It removes specified rows or columns.

# Q11. What is the purpose of abs()?
# Ans: It converts negative values into positive values.

# Q12. Why is redundancy removal important?
# Ans: It reduces unnecessary data and improves data quality.

