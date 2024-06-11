import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the cleaned data
df = pd.read_csv('data/cleaned/cleaned_data.csv')

# Display the first few rows
print(df.head())

# Summary statistics
print(df.describe())

# Correlation matrix
corr = df.corr()
sns.heatmap(corr, annot=True, fmt=".2f")
plt.show()

# Distribution of target variable
sns.histplot(df['price'], kde=True)
plt.show()

# Scatter plot of some features vs SalePrice
sns.scatterplot(data=df, x='sqft_living', y='price')
plt.show()
