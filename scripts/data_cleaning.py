import pandas as pd

# Load the raw data
df = pd.read_csv('../data/raw/usa_real_estate_dataset.csv')

# Select necessary columns
df = df[['price', 'bed', 'bath', 'acre_lot', 'house_size']]

# Data cleaning steps
df = df.dropna()  # Drop rows with missing values
df = df.drop_duplicates()  # Drop duplicate rows

# Save the cleaned data
df.to_csv('../data/cleaned/cleaned_data.csv', index=False)
