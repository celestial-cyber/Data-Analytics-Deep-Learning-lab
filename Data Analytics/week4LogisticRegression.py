# Logistic Regression is a supervised machine learning algorithm.
# It is used for classification problems.
# It predicts the probability of an outcome belonging to a class.
# Here, it predicts whether a student will Pass (1) or Fail (0).
# Features: Hours_Studied and Attendance.
# Target: Passed.


# Step 1: Import required libraries.
# Step 2: Create a sample student dataset.
# Step 3: Separate input features (X) and target variable (y).
# Step 4: Split the dataset into training and testing sets.
# Step 5: Create and train the Logistic Regression model.
# Step 6: Predict results using the test data.
# Step 7: Calculate accuracy, confusion matrix and classification report.
# Step 8: Visualize actual and predicted results.

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Create sample dataset
data = {
    'Hours_Studied': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
    'Attendance': [1, 0, 1, 1, 0, 0, 1, 1, 0, 1],
    'Passed': [0, 0, 1, 1, 0, 0, 1, 1, 0, 1]
}

df = pd.DataFrame(data)

# Define features and target
X = df[['Hours_Studied', 'Attendance']]
y = df['Passed']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))

# Visualize actual and predicted results
plt.scatter(df['Hours_Studied'], df['Passed'],
            color='blue', label='Actual')

plt.scatter(X_test['Hours_Studied'], y_pred,
            color='red', label='Predicted')

plt.xlabel('Hours Studied')
plt.ylabel('Passed')
plt.title('Logistic Regression Results')
plt.legend()
plt.show()

# Accuracy: 1.0
#
# Confusion Matrix:
# [[1 0]
#  [0 1]]
#
# Classification Report:
#              precision  recall  f1-score  support
#
#           0       1.00    1.00      1.00        1
#           1       1.00    1.00      1.00        1
#
# The scatter plot displays:
# Blue points - Actual results.
# Red points  - Predicted results.


# 1. Logistic Regression is used for classification.
# 2. It predicts the probability of a class.
# 3. X contains Hours_Studied and Attendance.
# 4. y contains the Passed column.
# 5. fit() trains the model.
# 6. predict() predicts the output.
# 7. accuracy_score() measures model accuracy.
# 8. confusion_matrix() shows correct and incorrect predictions.
# 9. classification_report() displays precision, recall and F1-score.

# Q1. What is Logistic Regression?
# Ans: It is a supervised learning algorithm used for classification.

# Q2. Is Logistic Regression used for classification or regression?
# Ans: It is mainly used for classification.

# Q3. What is the target variable in this program?
# Ans: Passed.

# Q4. What are the input features?
# Ans: Hours_Studied and Attendance.

# Q5. What is the meaning of 1 and 0?
# Ans: 1 means Passed and 0 means Failed.

# Q6. What is train_test_split()?
# Ans: It divides data into training and testing sets.

# Q7. What does fit() do?
# Ans: It trains the model using training data.

# Q8. What does predict() do?
# Ans: It predicts the output for new data.

# Q9. What is a confusion matrix?
# Ans: It shows correct and incorrect classification results.

# Q10. What is accuracy?
# Ans: It is the ratio of correct predictions to total predictions.

# Q11. What is the sigmoid function?
# Ans: It converts values into probabilities between 0 and 1.

# Q12. What is the difference between Linear and Logistic Regression?
# Ans: Linear Regression predicts continuous values,
#      while Logistic Regression predicts classes.

