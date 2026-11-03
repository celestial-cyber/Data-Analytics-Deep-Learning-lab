# Imputation is a technique used to handle missing values in a dataset.
# Mean Imputation replaces missing values with the average of the column.
# SimpleImputer is a class from sklearn.impute.
# Here, missing values in columns A and B are replaced with their mean.

# Step 1: Import pandas and SimpleImputer.
# Step 2: Create a sample DataFrame containing missing values.
# Step 3: Display the original DataFrame.
# Step 4: Create a SimpleImputer using mean strategy.
# Step 5: Apply fit_transform() to replace missing values.
# Step 6: Create a new DataFrame with the imputed values.
# Step 7: Display the updated DataFrame.

import pandas as pd
from sklearn.impute import SimpleImputer

# Create sample DataFrame
data = {
    'A': [1, 2, None, 4, 5],
    'B': [None, 2, 3, 4, None],
    'C': [1, 2, 3, 4, 5]
}

df = pd.DataFrame(data)

# Display original DataFrame
print("Original DataFrame:\n", df)

# Mean Imputation
imputer = SimpleImputer(strategy='mean')

df_imputed = pd.DataFrame(
    imputer.fit_transform(df),
    columns=df.columns
)

# Display imputed DataFrame
print("\nDataFrame After Mean Imputation:\n", df_imputed)

# Original DataFrame:
#      A    B  C
# 0  1.0  NaN  1
# 1  2.0  2.0  2
# 2  NaN  3.0  3
# 3  4.0  4.0  4
# 4  5.0  NaN  5
#
# DataFrame After Mean Imputation:
#      A    B    C
# 0  1.0  3.0  1.0
# 1  2.0  2.0  2.0
# 2  3.0  3.0  3.0
# 3  4.0  4.0  4.0
# 4  5.0  3.0  5.0

# 1. Imputation is used to handle missing values.
# 2. Mean Imputation replaces missing values with the column average.
# 3. SimpleImputer is available in sklearn.impute.
# 4. strategy='mean' performs mean imputation.
# 5. fit() calculates the mean of each column.
# 6. transform() replaces missing values with the calculated mean.
# 7. fit_transform() performs both operations together.
# 8. None values are treated as NaN in numeric DataFrames.

# Q1. What is imputation?
# Ans: Imputation is the process of replacing missing values in a dataset.

# Q2. What is Mean Imputation?
# Ans: It replaces missing values with the average of the column.

# Q3. Which library is used for imputation?
# Ans: Scikit-learn.

# Q4. What is SimpleImputer?
# Ans: It is a class used to handle missing values in datasets.

# Q5. What is strategy='mean'?
# Ans: It replaces missing values with the column mean.

# Q6. What does fit() do?
# Ans: It calculates the mean of each column.

# Q7. What does transform() do?
# Ans: It replaces missing values with the calculated mean.

# Q8. What is fit_transform()?
# Ans: It calculates the required values and replaces missing values.

# Q9. What is NaN?
# Ans: NaN means Not a Number and represents a missing value.

# Q10. Name other imputation strategies.
# Ans: Median, most_frequent and constant.

# Q11. What is the advantage of imputation?
# Ans: It helps retain useful data instead of deleting missing-value rows.
