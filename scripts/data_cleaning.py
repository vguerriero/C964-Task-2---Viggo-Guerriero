import pandas as pd

# Load the raw data
df = pd.read_csv('../data/raw/house_prices.csv')

# Data cleaning steps
df = df.drop(columns=['Alley', 'PoolQC', 'Fence', 'MiscFeature'])  # Drop columns with many missing values
df = df.dropna()  # Drop rows with missing values

# Save the cleaned data
df.to_csv('../data/cleaned/cleaned_data.csv', index=False)
