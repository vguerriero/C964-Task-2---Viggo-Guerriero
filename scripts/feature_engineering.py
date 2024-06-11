import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the cleaned data
df = pd.read_csv('../data/cleaned/cleaned_data.csv')

# Feature engineering example
df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']

# Standardize features
scaler = StandardScaler()
numerical_features = ['TotalSF', 'GrLivArea']
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Save the engineered data
df.to_csv('../data/cleaned/engineered_data.csv', index=False)
