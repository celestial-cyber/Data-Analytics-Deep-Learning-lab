from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

X = np.array([1,2,3,4,5]).reshape(-1,1)
Y = np.array([1.2,2.3,2.8,3.6,4.5])

model = LinearRegression()
model.fit(X,Y)

Y_pred = model.fit(X,Y)

plt.scatter(X,Y,color='blue')
plt.plot(Y_pred,color='red')
plt.show()

# reshape(-1, 1) converts X from a 1D array into a 2D array.
# -1 → NumPy automatically calculates the number of rows.
#  1 → specifies that there should be 1 column (1 feature).
# This gives X a shape of (5, 1), which is the format expected by ML models.

