import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import joblib

# Load the engineered data
df = pd.read_csv('data/cleaned/engineered_data.csv')

# Define features and target
features = ['sqft_living', 'bedrooms', 'bathrooms', 'floors', 'sqft_lot']
X = df[features]
y = df['price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model using GridSearchCV for hyperparameter tuning
xgb = XGBRegressor(random_state=42)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.2]
}

grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=2)
grid_search.fit(X_train, y_train)

# Get the best model from grid search
best_xgb = grid_search.best_estimator_

# Predict and evaluate the model
y_pred = best_xgb.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Save the best model
joblib.dump(best_xgb, 'models/house_price_xgb_model.pkl')
