# THEORY:
# Data visualization is the graphical representation of data.
# It helps to understand patterns, trends and relationships.
#
# Different visualization techniques include:
# 1. Bar Chart – Compares values across categories.
# 2. Column Chart – Represents categorical data using vertical bars.
# 3. Line Chart – Shows trends and changes over time or sequence.
# 4. Scatter Plot – Shows the relationship between two variables.
# 5. 3D Scatter Plot – Represents data using X, Y and Z axes.

# ALGORITHM:
# 1. Import Matplotlib and 3D plotting libraries.
# 2. Create sample data for categories and values.
# 3. Create X, Y and Z coordinates.
# 4. Plot a bar chart.
# 5. Plot a column chart.
# 6. Plot a line chart.
# 7. Plot a scatter plot.
# 8. Display all 2D charts using subplots.
# 9. Create a 3D scatter plot.
# 10. Display the 3D visualization.

import matplotlib.pyplot as plt

# Sample data
categories = ['A', 'B', 'C', 'D']
values = [10, 20, 15, 25]

x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 7, 11]
z = [1, 4, 9, 16, 25]

# Create 2D visualizations
plt.figure(figsize=(12, 8))

# Bar Chart
plt.subplot(2, 2, 1)
plt.barh(categories, values)
plt.title('Bar Chart')
plt.xlabel('Values')
plt.ylabel('Categories')

# Column Chart
plt.subplot(2, 2, 2)
plt.bar(categories, values)
plt.title('Column Chart')
plt.xlabel('Categories')
plt.ylabel('Values')

# Line Chart
plt.subplot(2, 2, 3)
plt.plot(x, y, 'ro-')
plt.title('Line Chart')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# Scatter Plot
plt.subplot(2, 2, 4)
plt.scatter(x, y)
plt.title('Scatter Plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

plt.tight_layout()
plt.show()

# 3D Scatter Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z)

ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title('3D Scatter Plot')

plt.show()


# EXPECTED OUTPUT:
#
# 1. Bar Chart:
#    Displays horizontal bars comparing categories A, B, C and D.
#
# 2. Column Chart:
#    Displays vertical bars comparing category values.
#
# 3. Line Chart:
#    Shows the relationship between X and Y values using a line.
#
# 4. Scatter Plot:
#    Displays individual points representing X and Y values.
#
# 5. 3D Scatter Plot:
#    Displays points in three dimensions using X, Y and Z axes.
#
# The first window contains four 2D charts.
# The second window displays the 3D scatter plot.

# IMPORTANT KEY POINTS:
#
# 1. Data Visualization:
#    Represents data graphically for easier understanding.
#
# 2. Matplotlib:
#    Python library used for creating graphs and charts.
#
# 3. Bar Chart:
#    Uses horizontal bars to compare categories.
#
# 4. Column Chart:
#    Uses vertical bars to compare values.
#
# 5. Line Chart:
#    Shows trends and changes in data.
#
# 6. Scatter Plot:
#    Shows relationships between two numerical variables.
#
# 7. 3D Scatter Plot:
#    Represents data using three dimensions.
#
# 8. subplot():
#    Creates multiple plots in a single figure.
#
# 9. scatter():
#    Creates a scatter plot.
#
# 10. projection='3d':
#     Creates a three-dimensional plotting area.
#
# 11. tight_layout():
#     Adjusts spacing between multiple plots.

