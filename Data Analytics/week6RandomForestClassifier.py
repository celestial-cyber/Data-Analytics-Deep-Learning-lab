# THEORY:
# Random Forest is a supervised machine learning algorithm
# used for classification and regression.
#
# It creates multiple decision trees and combines their
# predictions to produce the final result.
#
# In this program, Random Forest is used to classify
# Iris flowers into three species based on flower measurements.
#
# The model is evaluated using accuracy, confusion matrix
# and classification report.

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


# EXPECTED OUTPUT:
#
# Accuracy: 1.0
#
# Confusion Matrix:
# [[10  0  0]
#  [ 0  9  0]
#  [ 0  0 11]]
#
# Classification Report:
# Displays precision, recall, F1-score and support
# for the three Iris flower species.
#
# Note: The exact output depends on the dataset split
# and model configuration.


# VIVA QUESTIONS:
#
# 1. What is Random Forest?
#    Random Forest is a machine learning algorithm that
#    combines multiple decision trees.
#
# 2. What type of algorithm is Random Forest?
#    It is a supervised ensemble learning algorithm.
#
# 3. What is ensemble learning?
#    It combines multiple models to improve prediction
#    performance.
#
# 4. What is a decision tree?
#    It is a tree-based model that makes decisions
#    using conditions on input features.
#
# 5. What is the Iris dataset?
#    It is a dataset containing measurements of three
#    Iris flower species.
#
# 6. What is classification?
#    Classification is the process of predicting
#    categories or class labels.
#
# 7. What does n_estimators=100 mean?
#    It means the Random Forest contains 100 decision trees.
#
# 8. What is accuracy?
#    Accuracy is the ratio of correct predictions
#    to total predictions.
#
# 9. What is a confusion matrix?
#    It is a table that compares actual and predicted
#    class labels.
#
# 10. What is precision?
#     Precision measures how many predicted positive
#     cases are actually positive.
#
# 11. What is recall?
#     Recall measures how many actual positive cases
#     are correctly identified.
#
# 12. What is F1-score?
#     It is the harmonic mean of precision and recall.
#
# 13. What is the purpose of train_test_split()?
#     It divides the dataset into training and testing sets.
#
# 14. What is the advantage of Random Forest?
#     It reduces overfitting compared with a single
#     decision tree in many situations and improves
#     prediction stability.

