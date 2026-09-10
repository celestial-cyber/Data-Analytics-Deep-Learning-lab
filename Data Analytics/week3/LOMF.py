# ============================================================
# LAST OBSERVATION CARRIED FORWARD (LOCF)
# & NEXT OBSERVATION CARRIED BACKWARD (NOCB)
# ============================================================


# ============================================================
# THEORY
# ============================================================

# LOCF and NOCB are missing-value imputation methods
# commonly used for TIME SERIES DATA or LONGITUDINAL STUDIES.
#
# They are useful when the ORDER of observations is meaningful.
#
# For example:
#
# Date        Value
# 1 Jan        10
# 2 Jan        NaN
# 3 Jan        NaN
# 4 Jan        40
#
# Since the observations are arranged according to time,
# we can use the previous or next observed value to fill
# the missing values.


# ============================================================
# 1. LAST OBSERVATION CARRIED FORWARD (LOCF)
# ============================================================

# LOCF stands for:
# Last Observation Carried Forward.
#
# It replaces a missing value with the LAST OBSERVED VALUE
# before the missing value.
#
# Example:
#
# Original:
# [1, NaN, NaN, 4]
#
# After LOCF:
# [1,  1,   1,  4]
#
# The value 1 is carried forward to fill the missing values.
#
# LOCF is based on the assumption that the most recently
# observed value is the best estimate for the missing value.


# ============================================================
# 2. NEXT OBSERVATION CARRIED BACKWARD (NOCB)
# ============================================================

# NOCB stands for:
# Next Observation Carried Backward.
#
# It is the reverse of LOCF.
#
# It replaces a missing value with the NEXT OBSERVED VALUE
# after the missing value.
#
# Example:
#
# Original:
# [1, NaN, NaN, 4]
#
# After NOCB:
# [1,  4,   4,  4]
#
# The value 4 is carried backward to fill the missing values.


# ============================================================
# WHEN ARE LOCF AND NOCB USED?
# ============================================================

# LOCF and NOCB are mainly used when:
#
# 1. Data is time-series data.
# 2. The order of observations is important.
# 3. Observations are collected repeatedly over time.
# 4. The previous or next observation is considered
#    a reasonable estimate for the missing value.
#
# They are commonly seen in areas such as:
# - Clinical trials
# - Longitudinal studies
# - Patient monitoring
# - Sensor/time-series data


# ============================================================
# IMPORTANT LIMITATIONS
# ============================================================

# LOCF and NOCB are simple and easy to implement, but
# they can introduce BIAS into the data.
#
# They may underestimate the actual variability of the data.
#
# This can be especially problematic when:
#
# - The data has a strong trend over time.
# - The data has seasonality.
# - Missing values occur for long periods.
# - The missing values are not randomly distributed.
#
# Therefore, LOCF/NOCB should be selected according to
# the nature and context of the dataset.


# ============================================================
# PROGRAM
# ============================================================

import pandas as pd
import numpy as np


# ============================================================
# CREATE TIME-SERIES DATA WITH MISSING VALUES
# ============================================================

# pd.date_range() generates a sequence of dates.
#
# start='1/1/2023' → starting date is January 1, 2023
#
# periods=8 → generates 8 dates
#
# freq='D' → frequency is one day


time_data = {
    'Time': pd.date_range(
        start='1/1/2023',
        periods=8,
        freq='D'
    ),

    # np.nan represents missing values.
    #
    # Missing values are present on:
    # January 2
    # January 3
    # January 6

    'Value': [1, np.nan, np.nan, 4, 5, np.nan, 7, 8]
}


# Convert the dictionary into a Pandas DataFrame.

df_time = pd.DataFrame(time_data)


# ============================================================
# LOCF - LAST OBSERVATION CARRIED FORWARD
# ============================================================

# ffill() means Forward Fill.
#
# It carries the LAST AVAILABLE value forward
# to replace missing values.
#
# Therefore, ffill() is used to implement LOCF.

df_time['Value_LOCF'] = df_time['Value'].ffill()


# ============================================================
# NOCB - NEXT OBSERVATION CARRIED BACKWARD
# ============================================================

# bfill() means Backward Fill.
#
# It carries the NEXT AVAILABLE value backward
# to replace missing values.
#
# Therefore, bfill() is used to implement NOCB.

df_time['Value_NOCB'] = df_time['Value'].bfill()


# ============================================================
# DISPLAY THE RESULT
# ============================================================

print(df_time)


# ============================================================
# IMPORTANT FUNCTIONS / METHODS
# ============================================================

# pd.date_range()
# → Generates a sequence of dates at a specified frequency.
#
# Example:
# pd.date_range(start='1/1/2023', periods=8, freq='D')
#
# → Creates 8 consecutive daily dates.


# np.nan
# → Represents a missing value.


# pd.DataFrame()
# → Converts the given data into a tabular DataFrame.


# ffill()
# → Forward Fill.
# → Used for LOCF.
# → Replaces a missing value with the previous available value.


# bfill()
# → Backward Fill.
# → Used for NOCB.
# → Replaces a missing value with the next available value.


# ============================================================
# EXPECTED OUTPUT
# ============================================================

#         Time  Value  Value_LOCF  Value_NOCB
# 0  2023-01-01    1.0         1.0         1.0
# 1  2023-01-02    NaN         1.0         4.0
# 2  2023-01-03    NaN         1.0         4.0
# 3  2023-01-04    4.0         4.0         4.0
# 4  2023-01-05    5.0         5.0         5.0
# 5  2023-01-06    NaN         5.0         7.0
# 6  2023-01-07    7.0         7.0         7.0
# 7  2023-01-08    8.0         8.0         8.0


# ============================================================
# UNDERSTANDING THE OUTPUT
# ============================================================

# Original Value:
#
# [1, NaN, NaN, 4, 5, NaN, 7, 8]
#
#
# LOCF:
#
# [1, 1, 1, 4, 5, 5, 7, 8]
#
# Explanation:
#
# The first missing value comes after 1,
# so 1 is carried forward.
#
# The second missing value also comes after 1,
# so 1 is carried forward.
#
# The missing value after 5 is replaced by 5.


# NOCB:
#
# [1, 4, 4, 4, 5, 7, 7, 8]
#
# Explanation:
#
# The missing values before 4 are replaced by 4.
#
# The missing value before 7 is replaced by 7.


# ============================================================
# LOCF vs NOCB
# ============================================================

# LOCF:
# → Uses the PREVIOUS observed value.
# → Implemented using ffill().
# → Carries the value FORWARD.
#
#
# NOCB:
# → Uses the NEXT observed value.
# → Implemented using bfill().
# → Carries the value BACKWARD.


# ============================================================
# HOW TO CHOOSE AN IMPUTATION TECHNIQUE?
# ============================================================

# The choice of imputation technique depends on:
#
# 1. Type of data
# 2. Distribution of data
# 3. Pattern of missing values
# 4. Presence of outliers
# 5. Relationship between variables
# 6. Whether the data is time-dependent
#
#
# Mean imputation:
# → Suitable for numerical data without significant outliers.
#
# Median imputation:
# → Useful when data contains outliers or is skewed.
#
# Mode imputation:
# → Commonly used for categorical data.
#
# KNN imputation:
# → Uses similar observations to estimate missing values.
#
# LOCF:
# → Uses the previous observation.
# → Suitable when the previous value is a reasonable estimate.
#
# NOCB:
# → Uses the next observation.
# → Suitable when the next value is a reasonable estimate.


# ============================================================
# IMPORTANT CAUTION
# ============================================================

# Before selecting an imputation technique, perform
# Exploratory Data Analysis (EDA).
#
# EDA helps us understand:
#
# - Distribution of the data
# - Outliers
# - Trends
# - Seasonality
# - Missing-value patterns
#
# This helps us select an appropriate imputation technique.


# ============================================================
# CONCLUSION
# ============================================================

# LOCF and NOCB are simple missing-value imputation
# techniques mainly used for time-series and longitudinal data.
#
# LOCF uses the previous available observation,
# while NOCB uses the next available observation.
#
# In Pandas:
#
# LOCF → ffill()
# NOCB → bfill()
#
# These methods are easy to implement but may introduce
# bias when the data contains trends or seasonality.