import pandas as pd
import os

# Use absolute path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(base_dir, 'data/raw/data_house.csv')

# Load the raw data
df = pd.read_csv(csv_path)

# Select necessary columns for analysis
columns = [
    'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 
    'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 
    'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15'
]
df = df[columns]

# Data cleaning steps (if any)
df = df.dropna()  # Drop rows with missing values

# Save the cleaned data
cleaned_csv_path = os.path.join(base_dir, 'data/cleaned/cleaned_data.csv')
df.to_csv(cleaned_csv_path, index=False)
