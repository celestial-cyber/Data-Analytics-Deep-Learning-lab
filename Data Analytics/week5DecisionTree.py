# THEORY:
# Decision Tree is a supervised machine learning algorithm
# used for classification and regression.
#
# It represents decisions in a tree-like structure.
# It splits data based on feature conditions to predict
# the target class.
#
# In this program, Decision Tree Classification is used
# to classify Iris flowers into three species based on
# their measurements.
#
# The model is evaluated using accuracy, confusion matrix
# and classification report.

# ALGORITHM:
# 1. Import the required libraries.
# 2. Load the Iris dataset.
# 3. Separate input features (X) and target variable (y).
# 4. Split the dataset into training and testing sets.
# 5. Create a Decision Tree Classifier.
# 6. Train the model using model.fit().
# 7. Predict the test data using model.predict().
# 8. Calculate the accuracy score.
# 9. Display the confusion matrix and classification report.
# 10. Visualize the Decision Tree using plot_tree().

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Decision Tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Visualize Decision Tree
plt.figure(figsize=(15, 10))
plot_tree(
    model,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True
)
plt.title('Decision Tree for Iris Classification')
plt.show()


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
# Decision Tree Visualization:
# Displays a tree structure containing decision rules,
# feature conditions and predicted classes.
#
# Note: Exact output depends on the dataset and model settings.

# IMPORTANT KEY POINTS:
#
# 1. Decision Tree:
#    A supervised machine learning algorithm.
#
# 2. Decision Tree Induction:
#    The process of constructing a decision tree
#    from training data.
#
# 3. Root Node:
#    The topmost node of the decision tree.
#
# 4. Internal Node:
#    Represents a decision based on a feature.
#
# 5. Leaf Node:
#    Represents the final predicted class.
#
# 6. Iris Dataset:
#    Contains measurements of three Iris flower species.
#
# 7. Features (X):
#    Sepal length, sepal width, petal length and petal width.
#
# 8. Target (y):
#    Iris flower species.
#
# 9. model.fit():
#    Trains the Decision Tree model.
#
# 10. model.predict():
#     Predicts the class labels.
#
# 11. Accuracy:
#     Measures the percentage of correct predictions.
#
# 12. Confusion Matrix:
#     Compares actual and predicted class labels.
#
# 13. plot_tree():
#     Visualizes the structure of the Decision Tree.
#
# 14. filled=True:
#     Fills tree nodes with colors based on predicted classes.


# VIVA QUESTIONS:
#
# 1. What is a Decision Tree?
#    A Decision Tree is a supervised machine learning
#    algorithm used for classification and regression.
#
# 2. What is Decision Tree Induction?
#    It is the process of building a decision tree
#    from training data.
#
# 3. What are the main components of a Decision Tree?
#    Root node, internal nodes, branches and leaf nodes.
#
# 4. What is a root node?
#    It is the topmost node of the decision tree.
#
# 5. What is a leaf node?
#    It represents the final predicted class or output.
#
# 6. What is classification?
#    Classification predicts categories or class labels.
#
# 7. Which dataset is used in this program?
#    Iris dataset.
#
# 8. What are the features of the Iris dataset?
#    Sepal length, sepal width, petal length and petal width.
#
# 9. What is overfitting?
#    Overfitting occurs when a model learns training data
#    too closely and performs poorly on unseen data.
#
# 10. What is Gini Index?
#     It measures the impurity of data in a decision tree.
#
# 11. What is Entropy?
#     It measures the uncertainty or impurity in a dataset.
#
# 12. What is Information Gain?
#     It measures the reduction in entropy after a split.
#
# 13. What is the purpose of plot_tree()?
#     It displays the decision tree structure graphically.
#
# 14. What is the purpose of random_state=42?
#     It ensures reproducible results when splitting data
#     and training the model.
#
# 15. What is the advantage of a Decision Tree?
#     It is easy to understand, interpret and visualize.

