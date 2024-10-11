import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('index.csv')

print("Data Types:")
print(df.dtypes)

numeric_df = df.select_dtypes(include=[np.number])

if not numeric_df.empty:
    mean = numeric_df.mean()
    median = numeric_df.median()

    mode = numeric_df.mode().iloc[0]

    std_dev = numeric_df.std()
    variance = numeric_df.var()

    min_values = numeric_df.min()
    max_values = numeric_df.max()
    range_values = max_values - min_values

    percentiles = numeric_df.quantile([0.25, 0.5, 0.75])

df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
df['hour'] = df['datetime'].dt.hour
df['day'] = df['datetime'].dt.day
df['month'] = df['datetime'].dt.month
df['day_name'] = df['datetime'].dt.day_name()
days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
df['day_name'] = pd.Categorical(df['day_name'], categories=days_order, ordered=True)

df.head(10)

print("Mean: \n", mean)
print("Median: \n", median)
print("Mode: \n", mode)
print("Standard Deviation: \n", std_dev)
print("Variance: \n", variance)
print("Minimum: \n", min_values)
print("Maximum: \n", max_values)
print("Range: \n", range_values)
print("Percentiles: \n", percentiles)

coffee_names = df['coffee_name'].unique()
total_sale_by_coffee = df.groupby('coffee_name')['money'].sum()
count_sale_by_coffee = df.groupby('coffee_name')['money'].count()

sorted_total_sale = total_sale_by_coffee.sort_values()
sorted_count_sale = count_sale_by_coffee.loc[sorted_total_sale.index]

fig, ax1 = plt.subplots(figsize=(12, 8))

color = 'tab:blue'
ax1.set_xlabel('Coffee Name')
ax1.set_ylabel('Total Sales')
ax1.bar(sorted_total_sale.index, sorted_total_sale, color = color, alpha=0.6, label='Total Sales')
ax1.tick_params(axis='y', labelcolor=color)

plt.title('Total Sales by Coffee')
ax1.legend(loc='upper left')   
#ax1.legend(loc='upper right')

fig.tight_layout()
plt.show()

coffee_names = df['coffee_name'].unique()
total_sale_by_coffee = df.groupby('coffee_name')['money'].sum()
count_sale_by_coffee = df.groupby('coffee_name')['money'].count()

sorted_total_sale = total_sale_by_coffee.sort_values()
sorted_count_sale = count_sale_by_coffee.loc[sorted_total_sale.index]

fig, ax1 = plt.subplots(figsize=(12, 8))

ax2 = ax1.twinx()
color = 'tab:orange'
ax2.set_ylabel('Number of Sales', color=color)
ax2.plot(sorted_count_sale.index, sorted_count_sale, color=color, marker='o', label='Number of Sales')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Number of Sales by Coffee')  
ax2.legend(loc='upper left')

fig.tight_layout()
plt.show()

sale_by_coffee_perday = df.groupby(['day_name', 'coffee_name'])['money'].count().unstack().fillna(0)

# Plot sales by coffee per day using a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(sale_by_coffee_perday, cmap='coolwarm', annot=True, fmt='.0f', cbar_kws={'label': 'Number of Sales'})
plt.title('Sales Count by Coffee per Day')
plt.xlabel('Coffee Name')
plt.xticks(rotation=40)
plt.ylabel('Day of the Week')
plt.show()

payment_type = df['cash_type'].unique()
total_sale_by_payment = df.groupby('cash_type')['money'].sum()
count_sale_by_payment = df.groupby('cash_type')['money'].count()


plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.pie(total_sale_by_payment, labels=total_sale_by_payment.index, autopct='%1.1f%%', startangle=140, colors=['limegreen','pink'])
plt.title('Total Sales by Payment Type')


plt.subplot(1, 2, 2) 
plt.pie(count_sale_by_payment, labels=count_sale_by_payment.index, autopct='%1.1f%%', startangle=140, colors=['limegreen','pink'])
plt.title('Number of Sales by Payment Type')

plt.tight_layout()
plt.show()

df['cash_type'].hist()