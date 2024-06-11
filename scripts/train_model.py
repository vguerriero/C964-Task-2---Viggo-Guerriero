import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# Load dataset with features
df = pd.read_csv('../data/cleaned/cleaned_data_with_features.csv')

X = df.drop('price', axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print('Linear Regression MAE:', mean_absolute_error(y_test, y_pred_lr))
print('Linear Regression RMSE:', mean_squared_error(y_test, y_pred_lr, squared=False))

# Decision Tree
dt = DecisionTreeRegressor()
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
print('Decision Tree MAE:', mean_absolute_error(y_test, y_pred_dt))
print('Decision Tree RMSE:', mean_squared_error(y_test, y_pred_dt, squared=False))

# Random Forest
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print('Random Forest MAE:', mean_absolute_error(y_test, y_pred_rf))
print('Random Forest RMSE:', mean_squared_error(y_test, y_pred_rf, squared=False))

# Save the best model
joblib.dump(rf, '../models/house_price_model.pkl')

