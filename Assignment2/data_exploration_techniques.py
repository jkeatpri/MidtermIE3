import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('house_price_regression_dataset.csv')

print(df.isnull().sum())

df.dropna(inplace=True)

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

mean = df.mean()
median = df.median()

mode = df.mode().iloc[0]

std_dev = df.std()
variance = df.var()

min_values = df.min()
max_values = df.max()
range_values = max_values - min_values

percentiles = df.quantile([0.25, 0.5, 0.75])

print("Mean: \n", mean)
print("Median: \n", median)
print("Mode: \n", mode)
print("Standard Deviation: \n", std_dev)
print("Variance: \n", variance)
print("Minimum: \n", min_values)
print("Maximum: \n", max_values)
print("Range: \n", range)
print("Percentiles: \n", percentiles)

df.hist(bins=30, figsize=(10,8))
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6))
sns.boxplot(data=df)
plt.show()

correlation_matrix = df.corr()
plt.figure(figsize=(12,8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()