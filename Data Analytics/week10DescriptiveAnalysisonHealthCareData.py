# THEORY:
# Descriptive analytics is used to summarize and understand
# historical data using statistical methods and visualizations.
#
# In this program, healthcare data is analyzed to understand
# patient age, gender, blood pressure, cholesterol levels
# and diabetes status.
#
# Histograms and box plots are used to visualize data
# distributions and compare different patient groups.


# ALGORITHM:
# 1. Import the required libraries.
# 2. Create a sample healthcare dataset.
# 3. Convert the dataset into a Pandas DataFrame.
# 4. Display basic statistical information.
# 5. Calculate the gender count.
# 6. Calculate the diabetes status count.
# 7. Plot the age distribution using a histogram.
# 8. Plot blood pressure distribution by gender using a box plot.
# 9. Plot cholesterol levels by diabetes status using a box plot.


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create healthcare dataset
data = {
    'PatientID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Age': [25, 45, 35, 50, 65, 55, 40, 30, 70, 60],
    'Gender': ['F', 'M', 'F', 'F', 'M', 'M', 'F', 'M', 'F', 'M'],
    'BloodPressure': [120, 140, 130, 150, 160, 155, 135, 125, 165, 150],
    'Cholesterol': [200, 220, 215, 250, 240, 230, 210, 205, 255, 245],
    'Diabetes': ['No', 'Yes', 'No', 'No', 'Yes', 'Yes', 'No', 'No', 'Yes', 'Yes']
}

df = pd.DataFrame(data)

# Display basic statistics
print("Basic Statistics:")
print(df.describe())

# Count gender and diabetes status
print("\nGender Count:")
print(df['Gender'].value_counts())

print("\nDiabetes Count:")
print(df['Diabetes'].value_counts())

# Plot age distribution
sns.histplot(df['Age'], bins=10, kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.show()

# Blood pressure by gender
sns.boxplot(x='Gender', y='BloodPressure', data=df)
plt.title('Blood Pressure Distribution by Gender')
plt.show()

# Cholesterol by diabetes status
sns.boxplot(x='Diabetes', y='Cholesterol', data=df)
plt.title('Cholesterol Levels by Diabetes Status')
plt.show()

# EXPECTED OUTPUT:
#
# 1. Basic Statistics:
#    Displays count, mean, standard deviation, minimum,
#    maximum and quartile values for numerical columns.
#
# 2. Gender Count:
#    F    5
#    M    5
#
# 3. Diabetes Count:
#    No     5
#    Yes    5
#
# 4. First Graph:
#    Age Distribution – Histogram showing patient ages.
#
# 5. Second Graph:
#    Blood Pressure Distribution by Gender – Box plot
#    comparing blood pressure between male and female patients.
#
# 6. Third Graph:
#    Cholesterol Levels by Diabetes Status – Box plot
#    comparing cholesterol levels of diabetic and non-diabetic patients.


# IMPORTANT KEY POINTS:
#
# 1. Descriptive Analytics:
#    Summarizes and describes existing data.
#
# 2. Pandas:
#    Used for data handling and statistical analysis.
#
# 3. DataFrame:
#    A two-dimensional structure used to store healthcare data.
#
# 4. describe():
#    Displays statistical information such as mean, minimum,
#    maximum and standard deviation.
#
# 5. value_counts():
#    Counts the frequency of categorical values.
#
# 6. Histogram:
#    Shows the frequency distribution of numerical data.
#
# 7. Box Plot:
#    Displays median, quartiles, spread and possible outliers.
#
# 8. Seaborn:
#    Used to create statistical visualizations.
#
# 9. Matplotlib:
#    Used to display and customize graphs.
#
# 10. Healthcare Analytics:
#     Helps summarize patient information and identify
#     patterns in healthcare datasets.

