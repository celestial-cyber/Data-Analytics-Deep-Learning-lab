# ==============================
# IDENTIFYING DATA REDUNDANCY AND ELIMINATION
# ==============================

# CONCEPT OVERVIEW
# This program identifies duplicate (redundant) records in a dataset
# and removes them using the Pandas library.
#
# Data redundancy means storing the same information multiple times.
# Duplicate records increase storage requirements and may lead to
# incorrect analysis.

# WHY IS THIS CONCEPT USED?
# Duplicate data can:
# • Increase storage space.
# • Reduce data quality.
# • Produce incorrect analytical results.
# • Slow down data processing.
#
# Removing duplicates improves data accuracy and efficiency.

# REAL-WORLD APPLICATIONS
# • Student database management
# • Employee records
# • Customer databases
# • Banking systems
# • Hospital management systems
# • E-commerce websites

# THEORY
# Data redundancy occurs when the same record appears multiple times
# in a dataset.
#
# Pandas provides two useful functions:
#
# duplicated()
# Identifies duplicate rows and returns True for repeated records.
#
# drop_duplicates()
# Removes duplicate rows while keeping the first occurrence.

# ADVANTAGES
# • Improves data quality.
# • Saves storage space.
# • Prevents repeated records.
# • Makes analysis more accurate.
# • Easy to implement using Pandas.

# DISADVANTAGES
# • Important duplicate records may be removed if not checked.
# • Only exact duplicates are removed by default.
# • Large datasets require more memory.

# WHEN TO USE
# • Before data analysis.
# • Before training machine learning models.
# • During data cleaning.
# • While maintaining databases.

# WHEN NOT TO USE
# • When duplicate records are intentionally stored.
# • When every occurrence has different meaning.

# LINE-BY-LINE EXPLANATION

# Step 1:
# Import the Pandas library.
# Pandas is used for handling and analyzing tabular data.

# Step 2:
# Create a dictionary containing sample data.
# Here, "Alice" and "Bob" appear twice with the same age,
# making them duplicate records.

# Step 3:
# Convert the dictionary into a DataFrame.
# A DataFrame is a table consisting of rows and columns.

# Step 4:
# Display the original dataset.

# Step 5:
# Use duplicated() to identify duplicate rows.
# The first occurrence is considered original.
# Remaining identical rows are marked as duplicates.

# Step 6:
# Display only the duplicate rows.

# Step 7:
# Use drop_duplicates() to remove duplicate rows.
# Only the first occurrence of each duplicate record is retained.

# Step 8:
# Display the cleaned dataset after removing duplicates.

# DRY RUN

# Original Data

# Name    Age
# Alice   25
# Bob     30
# Alice   25
# David   22
# Eve     28
# Bob     30

# duplicated() Output

# False
# False
# True
# False
# False
# True

# Duplicate Rows

# Alice 25
# Bob 30

# After drop_duplicates()

# Alice 25
# Bob 30
# David 22
# Eve 28

# MEMORY VISUALIZATION

# data (Dictionary)
#        │
#        ▼
# DataFrame (df)
#
# Alice 25
# Bob 30
# Alice 25
# David 22
# Eve 28
# Bob 30
#
#        │
# duplicated()
#        ▼
#
# Duplicate Rows
#
# Alice 25
# Bob 30
#
#        │
# drop_duplicates()
#        ▼
#
# Cleaned DataFrame
#
# Alice 25
# Bob 30
# David 22
# Eve 28

# OUTPUT EXPLANATION
# duplicated() identifies repeated rows.
# Alice (25) and Bob (30) appear more than once.
# drop_duplicates() removes the repeated rows while keeping
# the first occurrence.

# TIME COMPLEXITY
# Best Case: O(n)
# Average Case: O(n)
# Worst Case: O(n)
#
# Reason:
# Each row is checked once for duplication.

# SPACE COMPLEXITY
# O(n)
#
# A new DataFrame is created after removing duplicates.

# ALTERNATIVE APPROACHES

# 1. duplicated()
# Detects duplicate rows.

# 2. drop_duplicates()
# Removes duplicate rows.

# 3. SQL DISTINCT
# Removes duplicate records from database tables.

# 4. Hash-Based Duplicate Detection
# Uses hashing to efficiently identify duplicates in large datasets.

# COMPARISON

# duplicated()
# • Detects duplicates.
# • Does not remove them.

# drop_duplicates()
# • Removes duplicate rows.
# • Keeps the first occurrence by default.

# SQL DISTINCT
# • Removes duplicates directly from query results.

# COMMON MISTAKES

# • Forgetting to assign the result of drop_duplicates().
# • Assuming duplicated() removes duplicates.
# • Not checking duplicates before deletion.
# • Ignoring duplicate values in selected columns.

# REAL-WORLD ANALOGY

# Imagine a school admission office receives multiple
# application forms from the same student.
#
# duplicated() identifies repeated applications.
# drop_duplicates() keeps only one application and removes the rest.

# VIVA QUESTIONS

# Q1. What is data redundancy?
# Data redundancy means storing the same information multiple times.

# Q2. What is a duplicate record?
# A duplicate record is an identical copy of another record.

# Q3. What does duplicated() do?
# It identifies duplicate rows in a dataset.

# Q4. What does drop_duplicates() do?
# It removes duplicate rows while keeping the first occurrence.

# Q5. Which library is used?
# Pandas.

# Q6. Why is removing redundancy important?
# It improves data quality, saves storage, and increases analysis accuracy.

# Q7. Does duplicated() remove duplicates?
# No. It only identifies them.

# EXAM TIPS

# • Define data redundancy.
# • Explain duplicate records.
# • Mention duplicated() and drop_duplicates().
# • State that the first occurrence is retained by default.
# • Write at least two real-world applications.

# SUMMARY

# • Data redundancy means duplicate information.
# • duplicated() identifies duplicate rows.
# • drop_duplicates() removes duplicate rows.
# • Removing redundancy improves data quality.
# • Commonly used during data preprocessing in Data Science and Machine Learning.

# Step 1: Import pandas
import pandas as pd

# Step 2: Create sample data with redundant (duplicate) rows
data = {
    'Name': ['Alice', 'Bob', 'Alice', 'David', 'Eve', 'Bob'],
    'Age': [25, 30, 25, 22, 28, 30]
}

# Step 3: Convert dictionary into a DataFrame
df = pd.DataFrame(data)

# Step 4: Display the original dataset
print("Original Data:")
print(df)

# Step 5: Identify duplicate rows
duplicates = df[df.duplicated()]

# Step 6: Display duplicate rows
print("\nDuplicate Rows:")
print(duplicates)

# Step 7: Remove duplicate rows
df_cleaned = df.drop_duplicates()

# Step 8: Display the cleaned dataset
print("\nData after removing duplicates:")
print(df_cleaned)