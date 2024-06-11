import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the cleaned data
df = pd.read_csv('data/cleaned/cleaned_data.csv')

# Feature engineering example
df['TotalSF'] = df['sqft_living'] + df['sqft_basement']

# Standardize features
scaler = StandardScaler()
numerical_features = ['TotalSF', 'sqft_living']
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Save the modified dataset
df.to_csv('data/cleaned/engineered_data.csv', index=False)
