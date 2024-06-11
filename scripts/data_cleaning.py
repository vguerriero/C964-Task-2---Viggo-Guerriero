import pandas as pd

# Load the raw data
df = pd.read_csv('data/raw/data_house.csv')

# Select necessary columns for analysis
columns = [
    'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 
    'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 
    'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15'
]
df = df[columns]

# Data cleaning steps
df = df.dropna()  # Drop rows with missing values

# Save the cleaned data
df.to_csv('data/cleaned/cleaned_data.csv', index=False)
