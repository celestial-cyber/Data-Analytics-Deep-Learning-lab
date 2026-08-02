# ==============================
# WEEK 8: VISUALIZATION TECHNIQUES
# EXPERIMENT 9: PERFORM VISUALIZATION TECHNIQUES
# (BAR, COLUMN, LINE, SCATTER, 3D BAR CHART)
# ==============================

# Step 1: Import necessary libraries
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ---------------------------------------------------
# CODE EXPLANATION
# ---------------------------------------------------

# Why do we import matplotlib.pyplot?
# Matplotlib is the most commonly used Python library
# for creating graphs and charts.

# Why do we import Axes3D?
# Axes3D enables creation of 3D graphs in Matplotlib.

# Step 2: Create sample data
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May']
sales = [100, 150, 120, 180, 200]

# Why use sample data?
# The data represents monthly sales and is used
# to demonstrate different visualization techniques.

# ---------------------------------------------------
# BAR CHART
# ---------------------------------------------------

# Step 3: Create Bar Chart
plt.figure(figsize=(5,4))
plt.bar(months, sales)

# Why use bar()?
# It displays data using vertical rectangular bars.
# It is useful for comparing values across categories.

plt.title("Bar Chart")
plt.xlabel("Months")
plt.ylabel("Sales")
plt.show()

# ---------------------------------------------------
# COLUMN CHART
# ---------------------------------------------------

# Note:
# In Matplotlib, Bar Chart and Column Chart are the same.
# Vertical bars represent a Column Chart.

plt.figure(figsize=(5,4))
plt.bar(months, sales, color='green')

plt.title("Column Chart")
plt.xlabel("Months")
plt.ylabel("Sales")
plt.show()

# ---------------------------------------------------
# LINE CHART
# ---------------------------------------------------

# Step 4: Create Line Chart
plt.figure(figsize=(5,4))
plt.plot(months, sales, marker='o')

# Why use plot()?
# It connects data points with lines.
# It is useful for showing trends over time.

plt.title("Line Chart")
plt.xlabel("Months")
plt.ylabel("Sales")
plt.show()

# ---------------------------------------------------
# SCATTER PLOT
# ---------------------------------------------------

# Step 5: Create Scatter Plot
plt.figure(figsize=(5,4))
plt.scatter(months, sales)

# Why use scatter()?
# Scatter Plot displays individual data points.
# It is useful for identifying patterns,
# relationships, and outliers.

plt.title("Scatter Plot")
plt.xlabel("Months")
plt.ylabel("Sales")
plt.show()

# ---------------------------------------------------
# 3D BAR CHART
# ---------------------------------------------------

# Step 6: Create 3D Figure
fig = plt.figure(figsize=(6,5))
ax = fig.add_subplot(111, projection='3d')

# Why use projection='3d'?
# It creates a three-dimensional plotting area.

# Step 7: Define coordinates
x = [0, 1, 2, 3, 4]
y = [0, 0, 0, 0, 0]
z = [0, 0, 0, 0, 0]

dx = [0.5] * 5
dy = [0.5] * 5
dz = sales

# Why define dx, dy, dz?
# dx = Width of bars
# dy = Depth of bars
# dz = Height of bars (sales values)

# Step 8: Draw 3D bars
ax.bar3d(x, y, z, dx, dy, dz)

# Why use bar3d()?
# It displays values as three-dimensional bars.

ax.set_title("3D Bar Chart")
ax.set_xlabel("Month Index")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Sales")

plt.show()

# ---------------------------------------------------
# SAMPLE OUTPUT
# ---------------------------------------------------

# Output 1:
# A Bar Chart displaying monthly sales.

# Output 2:
# A Column Chart displaying monthly sales.

# Output 3:
# A Line Chart connecting monthly sales values.

# Output 4:
# A Scatter Plot showing individual sales points.

# Output 5:
# A 3D Bar Chart representing sales in three dimensions.

# Example Data Used:
#
# Month   Sales
# ----------------
# Jan      100
# Feb      150
# Mar      120
# Apr      180
# May      200
#
# Note:
# Five separate graph windows will be displayed,
# each representing a different visualization technique.