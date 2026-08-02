#Week 3: Imputation Model
#Experiment 2: Implement Mean Imputation using SimpleImputer

# Step 1: Import necessary libraries
import pandas as pd
from sklearn.impute import SimpleImputer

# Step 2: Create sample dataset with missing values
data = {
    'Age': [25, 30, None, 28, 35, None, 40],
    'Salary': [50000, 60000, 55000, None, 70000, 65000, None]
}

# Step 3: Convert dictionary into DataFrame
df = pd.DataFrame(data)

# Step 4: Display original dataset
print("Original Data:")
print(df)

# Step 5: Create SimpleImputer object using mean strategy
imputer = SimpleImputer(strategy='mean')

# Step 6: Replace missing values with mean
df_imputed = pd.DataFrame(
    imputer.fit_transform(df),
    columns=df.columns
)

# Step 7: Display dataset after imputation
print("\nData after Mean Imputation:")
print(df_imputed)

# Original Data:

#     Age   Salary
# 0  25.0  50000.0
# 1  30.0  60000.0
# 2   NaN  55000.0
# 3  28.0      NaN
# 4  35.0  70000.0
# 5   NaN  65000.0
# 6  40.0      NaN

# Data after Mean Imputation:

#      Age      Salary
# 0  25.0   50000.0
# 1  30.0   60000.0
# 2  31.6   55000.0
# 3  28.0   60000.0
# 4  35.0   70000.0
# 5  31.6   65000.0
# 6  40.0   60000.0

# ==============================
# MEMORY VISUALIZATION
# ==============================

# Original DataFrame (df)
#
#        Age    Salary
#      ----------------
#      25     50000
#      30     60000
#      NaN    55000
#      28     NaN
#      35     70000
#      NaN    65000
#      40     NaN
#
#              │
#              │ fit()
#              ▼
#
# Mean(Age) = 31.6
# Mean(Salary) = 60000
#
#              │
#              │ transform()
#              ▼
#
# New DataFrame (df_imputed)
#
#        Age    Salary
#      ----------------
#      25.0   50000
#      30.0   60000
#      31.6   55000
#      28.0   60000
#      35.0   70000
#      31.6   65000
#      40.0   60000
#
# The original DataFrame remains unchanged.
# A new DataFrame containing the imputed values is created.

# ==============================
# TIME & SPACE COMPLEXITY
# ==============================

# Time Complexity
#
# Best Case    : O(n)
# Average Case : O(n)
# Worst Case   : O(n)
#
# Reason:
# The algorithm scans each column once to calculate the mean
# and once again to replace missing values.
# Therefore, the overall complexity is linear with respect
# to the number of data values.

# Space Complexity
#
# O(n)
#
# Reason:
# A new DataFrame is created to store the imputed values.

# ==============================
# ADVANTAGES
# ==============================

# 1. Simple and easy to implement.
# 2. Fast even for large datasets.
# 3. Prevents loss of data caused by deleting rows.
# 4. Preserves the number of observations.
# 5. Suitable for numerical datasets.
# 6. Supported directly by Scikit-learn.
# 7. Useful during Machine Learning preprocessing.

# ==============================
# DISADVANTAGES
# ==============================

# 1. Mean is affected by extreme values (outliers).
# 2. Reduces the natural variation of data.
# 3. Can introduce bias into the dataset.
# 4. Not suitable for categorical variables.
# 5. May reduce prediction accuracy if many values are missing.

# ==============================
# APPLICATIONS
# ==============================

# 1. Machine Learning data preprocessing.
# 2. Data cleaning before analysis.
# 3. Medical datasets with missing patient information.
# 4. Banking and financial data.
# 5. Survey data analysis.
# 6. Customer databases.
# 7. Educational datasets.
# 8. Sales and marketing analytics.
# 9. IoT sensor data preprocessing.
# 10. Research and statistical analysis.

# ==============================
# EXAM TIPS
# ==============================

# • Define imputation before writing the algorithm.
# • Mention that missing values can reduce model accuracy.
# • State that Mean Imputation replaces missing values with
#   the average of the column.
# • Mention that SimpleImputer belongs to sklearn.impute.
# • Explain the difference between fit() and transform().
# • Mention alternative strategies:
#      - mean
#      - median
#      - most_frequent
#      - constant
# • Mention one real-world application to score extra marks.
# • State that mean imputation should be used only for
#   numerical data.

# ==============================
# SUMMARY
# ==============================

# • Imputation is the process of replacing missing values.
# • Mean Imputation replaces missing numerical values with
#   the average of the column.
# • SimpleImputer is provided by Scikit-learn.
# • fit() calculates the mean.
# • transform() replaces missing values.
# • fit_transform() performs both operations together.
# • It is one of the most commonly used preprocessing
#   techniques in Machine Learning.
# • It prevents unnecessary loss of data.

# ==============================
# 20 VIVA QUESTIONS WITH ANSWERS
# ==============================

# Q1. What is imputation?
# Ans: Imputation is the process of replacing missing values
# with estimated values instead of deleting records.

# Q2. What is Mean Imputation?
# Ans: It replaces missing numerical values with the average
# of the respective column.

# Q3. Why do datasets contain missing values?
# Ans: Due to human errors, sensor failures, data corruption,
# incomplete surveys, or transmission errors.

# Q4. Which library is used for Mean Imputation?
# Ans: Scikit-learn (sklearn.impute).

# Q5. What is SimpleImputer?
# Ans: It is a Scikit-learn class used to fill missing values
# using different strategies.

# Q6. What are the available strategies in SimpleImputer?
# Ans: mean, median, most_frequent, and constant.

# Q7. Why is 'mean' used in this program?
# Ans: Because the dataset contains numerical values.

# Q8. Can Mean Imputation be used for text data?
# Ans: No. It is suitable only for numerical data.

# Q9. What does fit() do?
# Ans: It calculates the replacement value (mean).

# Q10. What does transform() do?
# Ans: It replaces missing values using the calculated mean.

# Q11. What is fit_transform()?
# Ans: It combines fit() and transform() into one operation.

# Q12. What is NaN?
# Ans: NaN stands for "Not a Number" and represents missing values.

# Q13. Why is Pandas used?
# Ans: To create and manipulate tabular data using DataFrames.

# Q14. Why is DataFrame preferred?
# Ans: Because it provides rows, columns, indexing, filtering,
# and compatibility with Machine Learning libraries.

# Q15. What happens if missing values are ignored?
# Ans: Machine Learning models may fail or produce poor predictions.

# Q16. What is the time complexity?
# Ans: O(n).

# Q17. What is the space complexity?
# Ans: O(n).

# Q18. What is one disadvantage of Mean Imputation?
# Ans: It is affected by outliers.

# Q19. What is one advantage of Mean Imputation?
# Ans: It is simple, fast, and preserves the number of records.

# Q20. Where is Mean Imputation used in real life?
# Ans: Machine Learning, healthcare, banking, survey analysis,
# IoT data preprocessing, and business analytics.