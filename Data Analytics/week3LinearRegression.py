# Linear Regression is a supervised machine learning algorithm.
# It is used to predict continuous numerical values.
# It establishes a linear relationship between input features and target.
# Here, Feature1 and Feature2 are used to predict the Target value.
# The model learns the relationship from training data.

# Step 1: Import the required libraries.
# Step 2: Create a sample dataset.
# Step 3: Separate input features (X) and target variable (y).
# Step 4: Split the dataset into training and testing sets.
# Step 5: Create and train the Linear Regression model.
# Step 6: Predict target values using test data.
# Step 7: Calculate Mean Squared Error and R-squared score.
# Step 8: Visualize actual and predicted results.

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Create sample dataset
data = {
    'Feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Feature2': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
    'Target': [2.5, 3.5, 6, 7, 8.5, 10, 11.5, 13, 14.5, 16]
}

df = pd.DataFrame(data)

# Define features and target
X = df[['Feature1', 'Feature2']]
y = df['Target']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Visualize actual and predicted results
plt.scatter(X_test['Feature1'], y_test,
            color='blue', label='Actual')

plt.scatter(X_test['Feature1'], y_pred,
            color='red', label='Predicted')

plt.xlabel('Feature1')
plt.ylabel('Target')
plt.title('Linear Regression Results')
plt.legend()
plt.show()

# Mean Squared Error: Approximately 0.0
# R-squared: 1.0
#
# The scatter plot displays:
# Blue points - Actual target values.
# Red points  - Predicted target values.
#
# Both sets of points overlap because the data follows
# an almost perfect linear relationship.


# Q1. What is Linear Regression?
# Ans: It is a supervised learning algorithm used to predict continuous values.

# Q2. What is the main purpose of Linear Regression?
# Ans: To find the relationship between input features and target values.

# Q3. What are the features in this program?
# Ans: Feature1 and Feature2.

# Q4. What is the target variable?
# Ans: Target.

# Q5. What is the difference between dependent and independent variables?
# Ans: Independent variables are inputs, while the dependent variable is the output.

# Q6. What does fit() do?
# Ans: It trains the model using training data.

# Q7. What does predict() do?
# Ans: It predicts output values for test data.

# Q8. What is Mean Squared Error?
# Ans: It is the average of squared differences between actual and predicted values.

# Q9. What is R-squared?
# Ans: It measures how well the model fits the data.

# Q10. What is the range of R-squared?
# Ans: It is usually between 0 and 1 for models with an intercept,
#      but it can also be negative on test data.

# Q11. What is train_test_split()?
# Ans: It divides data into training and testing sets.

# Q12. What is the difference between Linear and Logistic Regression?
# Ans: Linear Regression predicts continuous values,
#      while Logistic Regression predicts classes.

