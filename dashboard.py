import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd
from dash.dependencies import Input, Output, State
from joblib import load
import logging
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = load('models/house_price_xgb_model.pkl')

# Load the engineered data
df = pd.read_csv('data/cleaned/engineered_data.csv')

# Initialize the Dash app
app = dash.Dash(__name__)

# Set up logging
logging.basicConfig(filename='dashboard.log', level=logging.DEBUG, 
                    format='%(asctime)s %(levelname)s: %(message)s')

# Layout of the dashboard
app.layout = html.Div([
    html.H1("House Price Prediction Dashboard"),
    dcc.Graph(id='scatter-plot'),
    dcc.Graph(id='feature-importance'),
    html.Div([
        html.Label('Square Feet Living'),
        dcc.Input(id='sqft-living', type='number', value=2000, debounce=True),
        html.Label('Bedrooms'),
        dcc.Input(id='bedrooms', type='number', value=3, debounce=True),
        html.Label('Bathrooms'),
        dcc.Input(id='bathrooms', type='number', value=2, debounce=True),
        html.Label('Floors'),
        dcc.Input(id='floors', type='number', value=1, debounce=True),
        html.Label('Square Feet Lot'),
        dcc.Input(id='sqft-lot', type='number', value=5000, debounce=True),
        html.Button('Add Entry', id='add-entry-button', n_clicks=0),
        html.Button('Predict', id='predict-button', n_clicks=0),
        html.Div(id='input-entries'),
        html.Div(id='prediction-output')
    ])
])

# Storage for multiple input entries
input_entries = []

# Callback to update scatter plot
@app.callback(
    Output('scatter-plot', 'figure'),
    Input('sqft-living', 'value')
)
def update_scatter_plot(sqft_living):
    fig = px.scatter(df, x='sqft_living', y='price', trendline="ols", title="Price vs. Sqft Living")
    logging.info(f"Scatter plot updated for Sqft Living: {sqft_living}")
    return fig

# Callback to update feature importance plot
@app.callback(
    Output('feature-importance', 'figure'),
    Input('sqft-living', 'value')
)
def update_feature_importance(sqft_living):
    importance = model.feature_importances_
    feature_names = ['sqft_living', 'bedrooms', 'bathrooms', 'floors', 'sqft_lot']
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
    fig = px.bar(feature_importance_df, x='Feature', y='Importance', title='Feature Importance')
    logging.info(f"Feature importance plot updated for Sqft Living: {sqft_living}")
    return fig

# Callback to add input entries
@app.callback(
    Output('input-entries', 'children'),
    Input('add-entry-button', 'n_clicks'),
    State('bedrooms', 'value'),
    State('bathrooms', 'value'),
    State('sqft-living', 'value'),
    State('sqft-lot', 'value'),
    State('floors', 'value')
)
def add_entry(n_clicks, bedrooms, bathrooms, sqft_living, sqft_lot, floors):
    if n_clicks > 0:
        entry = {'bedrooms': bedrooms, 'bathrooms': bathrooms, 'sqft_living': sqft_living, 'sqft_lot': sqft_lot, 'floors': floors}
        input_entries.append(entry)
        logging.info(f"New entry added: {entry}")
        return html.Ul([html.Li(f"Bedrooms: {e['bedrooms']}, Bathrooms: {e['bathrooms']}, Sqft Living: {e['sqft_living']}, Sqft Lot: {e['sqft_lot']}, Floors: {e['floors']}") for e in input_entries])
    return ''

# Callback to make predictions
@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks')
)
def predict(n_clicks):
    if n_clicks > 0 and input_entries:
        # Prepare the features in the same order as expected by the model
        features = pd.DataFrame(input_entries, columns=['sqft_living', 'bedrooms', 'bathrooms', 'floors', 'sqft_lot'])
        logging.debug(f"Features DataFrame:\n{features}")

        # Standardize features as done during training
        scaler = StandardScaler()
        features = scaler.fit_transform(features)

        predictions = model.predict(features)
        logging.debug(f"Predictions: {predictions}")

        results = [f'Predicted House Price: ${p:,.2f}' for p in predictions]
        logging.info(f"Predictions made: {results}")
        return html.Ul([html.Li(result) for result in results])
    return ''

if __name__ == '__main__':
    app.run_server(debug=True)
