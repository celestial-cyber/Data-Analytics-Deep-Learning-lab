# ============================================================
# 4. IMPLEMENT LOGISTIC REGRESSION
# Student Pass/Fail Prediction
# ============================================================

# Import NumPy for numerical operations
import numpy as np

# Import Pandas for creating and handling the dataset
import pandas as pd

# Import Matplotlib for visualization
import matplotlib.pyplot as plt

# Import Seaborn for creating the confusion matrix heatmap
import seaborn as sns

# train_test_split → divides the dataset into training and testing data
from sklearn.model_selection import train_test_split

# LogisticRegression → creates the Logistic Regression model
from sklearn.linear_model import LogisticRegression

# Evaluation functions
#
# accuracy_score → calculates the percentage of correct predictions
# confusion_matrix → shows correct and incorrect classifications
# classification_report → gives precision, recall, F1-score and support
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ============================================================
# PROGRAM DESCRIPTION
# ============================================================

# Logistic Regression is a SUPERVISED MACHINE LEARNING algorithm
# mainly used for CLASSIFICATION problems.
#
# Here, we want to predict whether a student will PASS or FAIL
# based on the number of HOURS STUDIED.
#
# Target:
# Passed
#
# 0 → Failed
# 1 → Passed
#
# Logistic Regression predicts the PROBABILITY of belonging
# to a class and then assigns the final class based on a
# decision threshold (usually 0.5).


# ============================================================
# 1. CREATE SAMPLE DATASET
# ============================================================

data = {
    'Hours_Studied': [
        2, 4, 5, 7, 8, 10, 12, 14,
        16, 18, 20, 22, 24, 26, 28
    ],

    'Passed': [
        0, 0, 0, 0, 1, 0, 1, 1,
        1, 1, 1, 1, 1, 1, 1
    ]
}

# pd.DataFrame() converts the dictionary into a table
df = pd.DataFrame(data)

# Display the dataset
print("Dataset:\n", df)


# ============================================================
# 2. DEFINE FEATURES AND TARGET
# ============================================================

# X = independent/input variable
#
# Here, Hours_Studied is the feature used to predict
# whether the student passes or fails.

X = df[['Hours_Studied']]

# y = dependent/target variable
#
# 0 → Failed
# 1 → Passed

y = df['Passed']


# ============================================================
# 3. TRAIN-TEST SPLIT
# ============================================================

# train_test_split() divides the dataset into:
#
# Training data → used to train the model
# Testing data  → used to evaluate the model
#
# test_size=0.3 means 30% of the data is used for testing
# and 70% is used for training.
#
# random_state=42 ensures the same split every time.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)


# ============================================================
# 4. TRAIN LOGISTIC REGRESSION MODEL
# ============================================================

# LogisticRegression() creates the Logistic Regression model
model = LogisticRegression()

# fit() trains the model using the training data.
#
# The model learns the relationship between:
#
# Hours Studied → Probability of Passing

model.fit(X_train, y_train)


# ============================================================
# 5. MAKE PREDICTIONS
# ============================================================

# predict() predicts the CLASS (0 or 1)
# for the testing data.

y_pred = model.predict(X_test)


# ============================================================
# 6. MODEL EVALUATION
# ============================================================


# ------------------------------------------------------------
# ACCURACY
# ------------------------------------------------------------

# accuracy_score() calculates the percentage of
# predictions that are correct.
#
# Formula:
#
# Accuracy = Correct Predictions / Total Predictions
#
# Higher accuracy → better performance.

accuracy = accuracy_score(y_test, y_pred)


# ------------------------------------------------------------
# CONFUSION MATRIX
# ------------------------------------------------------------

# confusion_matrix() creates a table showing:
#
# True Negative  → Correctly predicted Failed
# True Positive  → Correctly predicted Passed
# False Positive → Failed predicted as Passed
# False Negative → Passed predicted as Failed

conf_matrix = confusion_matrix(y_test, y_pred)


# ------------------------------------------------------------
# CLASSIFICATION REPORT
# ------------------------------------------------------------

# classification_report() provides:
#
# Precision → How many predicted positives were actually positive
# Recall    → How many actual positives were correctly identified
# F1-score  → Harmonic mean of Precision and Recall
# Support   → Number of actual samples in each class

# zero_division=0 prevents an error/warning if a class
# has no predicted samples.

report = classification_report(
    y_test,
    y_pred,
    zero_division=0
)


# Display evaluation results
print("Accuracy:", accuracy)

print("\nConfusion Matrix:\n", conf_matrix)

print("\nClassification Report:\n", report)


# ============================================================
# 7. VISUALIZATION - CONFUSION MATRIX HEATMAP
# ============================================================

# Create a figure for the confusion matrix
plt.figure(figsize=(5, 4))

# sns.heatmap() creates a heatmap representation
# of the confusion matrix.
#
# annot=True → displays the numbers inside the cells
# fmt='d'    → displays values as integers
# cmap       → specifies the color scheme
#
# xticklabels → labels for predicted classes
# yticklabels → labels for actual classes

sns.heatmap(
    conf_matrix,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Failed', 'Passed'],
    yticklabels=['Failed', 'Passed']
)

# X-axis represents predicted values
plt.xlabel('Predicted')

# Y-axis represents actual values
plt.ylabel('Actual')

# Add title
plt.title('Confusion Matrix')

# Automatically adjust the layout
plt.tight_layout()

# Save the graph as an image
plt.savefig('confusion_matrix.png', dpi=150)

# Display the graph
plt.show()


# ============================================================
# 8. VISUALIZATION - LOGISTIC REGRESSION CURVE
# ============================================================

# Create a new figure
plt.figure(figsize=(6, 4))

# Display the actual data points
#
# Hours studied → X-axis
# Passed (0/1)  → Y-axis

plt.scatter(
    df['Hours_Studied'],
    df['Passed'],
    color='blue',
    label='Actual Data'
)


# np.linspace() generates 300 equally spaced values
# between the minimum and maximum hours studied.
#
# reshape(-1, 1):
# -1 → NumPy automatically calculates the number of rows
#  1 → creates one column because we have one feature

x_range = np.linspace(
    df['Hours_Studied'].min(),
    df['Hours_Studied'].max(),
    300
).reshape(-1, 1)


# predict_proba() returns the probability of each class.
#
# It returns two probabilities:
#
# [:, 0] → probability of Failed (0)
# [:, 1] → probability of Passed (1)
#
# We select [:, 1] because we want the probability
# that the student will PASS.

y_prob = model.predict_proba(x_range)[:, 1]


# Plot the predicted probability curve
plt.plot(
    x_range,
    y_prob,
    color='red',
    label='Predicted Probability'
)


# Label the X-axis
plt.xlabel('Hours Studied')

# Label the Y-axis
plt.ylabel('Passed (0/1)')

# Add title
plt.title('Logistic Regression Curve')

# Display legend
plt.legend()

# Adjust layout
plt.tight_layout()

# Display the graph
plt.show()


# ============================================================
# 4. IMPLEMENT LOGISTIC REGRESSION
# Student Pass/Fail Prediction
# ============================================================

# Import NumPy for numerical operations
import numpy as np

# Import Pandas for creating and handling the dataset
import pandas as pd

# Import Matplotlib for visualization
import matplotlib.pyplot as plt

# Import Seaborn for creating the confusion matrix heatmap
import seaborn as sns

# train_test_split → divides the dataset into training and testing data
from sklearn.model_selection import train_test_split

# LogisticRegression → creates the Logistic Regression model
from sklearn.linear_model import LogisticRegression

# Evaluation functions
#
# accuracy_score → calculates the percentage of correct predictions
# confusion_matrix → shows correct and incorrect classifications
# classification_report → gives precision, recall, F1-score and support
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ============================================================
# PROGRAM DESCRIPTION
# ============================================================

# Logistic Regression is a SUPERVISED MACHINE LEARNING algorithm
# mainly used for CLASSIFICATION problems.
#
# Here, we want to predict whether a student will PASS or FAIL
# based on the number of HOURS STUDIED.
#
# Target:
# Passed
#
# 0 → Failed
# 1 → Passed
#
# Logistic Regression predicts the PROBABILITY of belonging
# to a class and then assigns the final class based on a
# decision threshold (usually 0.5).


# ============================================================
# 1. CREATE SAMPLE DATASET
# ============================================================

data = {
    'Hours_Studied': [
        2, 4, 5, 7, 8, 10, 12, 14,
        16, 18, 20, 22, 24, 26, 28
    ],

    'Passed': [
        0, 0, 0, 0, 1, 0, 1, 1,
        1, 1, 1, 1, 1, 1, 1
    ]
}

# pd.DataFrame() converts the dictionary into a table
df = pd.DataFrame(data)

# Display the dataset
print("Dataset:\n", df)


# ============================================================
# 2. DEFINE FEATURES AND TARGET
# ============================================================

# X = independent/input variable
#
# Here, Hours_Studied is the feature used to predict
# whether the student passes or fails.

X = df[['Hours_Studied']]

# y = dependent/target variable
#
# 0 → Failed
# 1 → Passed

y = df['Passed']


# ============================================================
# 3. TRAIN-TEST SPLIT
# ============================================================

# train_test_split() divides the dataset into:
#
# Training data → used to train the model
# Testing data  → used to evaluate the model
#
# test_size=0.3 means 30% of the data is used for testing
# and 70% is used for training.
#
# random_state=42 ensures the same split every time.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)


# ============================================================
# 4. TRAIN LOGISTIC REGRESSION MODEL
# ============================================================

# LogisticRegression() creates the Logistic Regression model
model = LogisticRegression()

# fit() trains the model using the training data.
#
# The model learns the relationship between:
#
# Hours Studied → Probability of Passing

model.fit(X_train, y_train)


# ============================================================
# 5. MAKE PREDICTIONS
# ============================================================

# predict() predicts the CLASS (0 or 1)
# for the testing data.

y_pred = model.predict(X_test)


# ============================================================
# 6. MODEL EVALUATION
# ============================================================


# ------------------------------------------------------------
# ACCURACY
# ------------------------------------------------------------

# accuracy_score() calculates the percentage of
# predictions that are correct.
#
# Formula:
#
# Accuracy = Correct Predictions / Total Predictions
#
# Higher accuracy → better performance.

accuracy = accuracy_score(y_test, y_pred)


# ------------------------------------------------------------
# CONFUSION MATRIX
# ------------------------------------------------------------

# confusion_matrix() creates a table showing:
#
# True Negative  → Correctly predicted Failed
# True Positive  → Correctly predicted Passed
# False Positive → Failed predicted as Passed
# False Negative → Passed predicted as Failed

conf_matrix = confusion_matrix(y_test, y_pred)


# ------------------------------------------------------------
# CLASSIFICATION REPORT
# ------------------------------------------------------------

# classification_report() provides:
#
# Precision → How many predicted positives were actually positive
# Recall    → How many actual positives were correctly identified
# F1-score  → Harmonic mean of Precision and Recall
# Support   → Number of actual samples in each class

# zero_division=0 prevents an error/warning if a class
# has no predicted samples.

report = classification_report(
    y_test,
    y_pred,
    zero_division=0
)


# Display evaluation results
print("Accuracy:", accuracy)

print("\nConfusion Matrix:\n", conf_matrix)

print("\nClassification Report:\n", report)


# ============================================================
# 7. VISUALIZATION - CONFUSION MATRIX HEATMAP
# ============================================================

# Create a figure for the confusion matrix
plt.figure(figsize=(5, 4))

# sns.heatmap() creates a heatmap representation
# of the confusion matrix.
#
# annot=True → displays the numbers inside the cells
# fmt='d'    → displays values as integers
# cmap       → specifies the color scheme
#
# xticklabels → labels for predicted classes
# yticklabels → labels for actual classes

sns.heatmap(
    conf_matrix,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Failed', 'Passed'],
    yticklabels=['Failed', 'Passed']
)

# X-axis represents predicted values
plt.xlabel('Predicted')

# Y-axis represents actual values
plt.ylabel('Actual')

# Add title
plt.title('Confusion Matrix')

# Automatically adjust the layout
plt.tight_layout()

# Save the graph as an image
plt.savefig('confusion_matrix.png', dpi=150)

# Display the graph
plt.show()


# ============================================================
# 8. VISUALIZATION - LOGISTIC REGRESSION CURVE
# ============================================================

# Create a new figure
plt.figure(figsize=(6, 4))

# Display the actual data points
#
# Hours studied → X-axis
# Passed (0/1)  → Y-axis

plt.scatter(
    df['Hours_Studied'],
    df['Passed'],
    color='blue',
    label='Actual Data'
)


# np.linspace() generates 300 equally spaced values
# between the minimum and maximum hours studied.
#
# reshape(-1, 1):
# -1 → NumPy automatically calculates the number of rows
#  1 → creates one column because we have one feature

x_range = np.linspace(
    df['Hours_Studied'].min(),
    df['Hours_Studied'].max(),
    300
).reshape(-1, 1)


# predict_proba() returns the probability of each class.
#
# It returns two probabilities:
#
# [:, 0] → probability of Failed (0)
# [:, 1] → probability of Passed (1)
#
# We select [:, 1] because we want the probability
# that the student will PASS.

y_prob = model.predict_proba(x_range)[:, 1]


# Plot the predicted probability curve
plt.plot(
    x_range,
    y_prob,
    color='red',
    label='Predicted Probability'
)


# Label the X-axis
plt.xlabel('Hours Studied')

# Label the Y-axis
plt.ylabel('Passed (0/1)')

# Add title
plt.title('Logistic Regression Curve')

# Display legend
plt.legend()

# Adjust layout
plt.tight_layout()

# Display the graph
plt.show()

# ============================================================
# EVALUATION METRICS
# ============================================================

# ACCURACY:
# Measures the percentage of total predictions that are correct.
#
# Accuracy = Correct Predictions / Total Predictions
#
# Higher accuracy is generally better.


# CONFUSION MATRIX:
#
#                    Predicted
#                  Failed  Passed
#
# Actual Failed      TN      FP
# Actual Passed      FN      TP
#
# TN = True Negative
# FP = False Positive
# FN = False Negative
# TP = True Positive


# PRECISION:
# Out of all students predicted as Passed,
# how many actually Passed?
#
# Precision = TP / (TP + FP)


# RECALL:
# Out of all students who actually Passed,
# how many were correctly predicted?
#
# Recall = TP / (TP + FN)


# F1-SCORE:
# Harmonic mean of Precision and Recall.
#
# F1 = 2 × (Precision × Recall) / (Precision + Recall)


# SUPPORT:
# Number of actual samples belonging to each class.