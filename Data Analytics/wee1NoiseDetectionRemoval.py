# Noise refers to incorrect, unusual or unwanted values in a dataset.
# Noise detection identifies unusual data points called outliers.
# Isolation Forest is a machine learning algorithm used for anomaly detection.
# It identifies unusual observations by isolating them from normal data.
# fit_predict() detects anomalies and assigns labels:
#  1  = Normal data
# -1  = Anomaly or noise
# The detected noise is removed to obtain clean data.

# Step 1: Import pandas, numpy and IsolationForest.
# Step 2: Create a sample dataset containing noise.
# Step 3: Display the original DataFrame.
# Step 4: Create an Isolation Forest model.
# Step 5: Apply fit_predict() to detect anomalies.
# Step 6: Identify normal data points using label 1.
# Step 7: Remove the anomaly column.
# Step 8: Display the cleaned DataFrame.


import pandas as pd
from sklearn.ensemble import IsolationForest

# Create sample data with noise
data = {
    'A': [1, 2, 3, 4, 100, 6, 7, 8, 9, 10],
    'B': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
}

df = pd.DataFrame(data)

print("Original DataFrame:")
print(df)

# Detect noise using Isolation Forest
model = IsolationForest(contamination=0.1, random_state=42)

df['anomaly'] = model.fit_predict(df[['A', 'B']])

# Remove noise
df_clean = df[df['anomaly'] == 1].drop(columns=['anomaly'])

print("\nDataFrame after noise removal:")
print(df_clean)

# Original DataFrame:
#      A   B
# 0    1  10
# 1    2   9
# 2    3   8
# 3    4   7
# 4  100   6
# 5    6   5
# 6    7   4
# 7    8   3
# 8    9   2
# 9   10   1
#
# DataFrame after noise removal:
#     A   B
# 0   1  10
# 1   2   9
# 2   3   8
# 3   4   7
# 5   6   5
# 6   7   4
# 7   8   3
# 8   9   2
# 9  10   1
#
# Note: Exact anomaly detection may vary with model settings
# and scikit-learn version.

# 1. Noise means unwanted or incorrect data.
# 2. An outlier is a value that differs significantly from other observations.
# 3. Isolation Forest is used for anomaly detection.
# 4. contamination specifies the expected proportion of anomalies.
# 5. fit_predict() trains the model and predicts anomaly labels.
# 6. Label 1 represents normal observations.
# 7. Label -1 represents anomalies.
# 8. drop() removes unwanted columns.
# 9. Noise removal improves data quality.

# Q1. What is noise in a dataset?
# Ans: Noise refers to unwanted or incorrect data values.

# Q2. What is an outlier?
# Ans: An outlier is a data point that differs significantly from other values.

# Q3. What is noise detection?
# Ans: It is the process of identifying unusual or incorrect data points.

# Q4. Which algorithm is used in this program?
# Ans: Isolation Forest.

# Q5. What is Isolation Forest?
# Ans: It is an unsupervised machine learning algorithm used for anomaly detection.

# Q6. What is contamination?
# Ans: It specifies the expected proportion of anomalies in the dataset.

# Q7. What does fit_predict() do?
# Ans: It fits the model and predicts whether each data point is normal or anomalous.

# Q8. What does label 1 represent?
# Ans: Normal data.

# Q9. What does label -1 represent?
# Ans: Anomaly or noise.

# Q10. Why is noise removal important?
# Ans: It improves data quality and helps prevent unusual values
#      from affecting analysis.

# Q11. Is Isolation Forest supervised or unsupervised?
# Ans: It is an unsupervised learning algorithm.

# Q12. What is the purpose of random_state?
# Ans: It helps produce reproducible results.

