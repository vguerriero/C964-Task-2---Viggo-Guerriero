import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd
from dash.dependencies import Input, Output
from joblib import load

# Load the trained model
model = load('models/house_price_xgb_model.pkl')

# Load the engineered data
df = pd.read_csv('data/cleaned/engineered_data.csv')

# Initialize the Dash app
app = dash.Dash(__name__)

# Layout of the dashboard
app.layout = html.Div([
    html.H1("House Price Prediction Dashboard"),
    dcc.Graph(id='scatter-plot'),
    dcc.Graph(id='feature-importance'),
    html.Div([
        html.Label('Total SF'),
        dcc.Input(id='total-sf', type='number', value=1000),
        html.Label('Square Feet Living'),
        dcc.Input(id='sqft-living', type='number', value=1000),
        html.Label('Bedrooms'),
        dcc.Input(id='bedrooms', type='number', value=3),
        html.Label('Bathrooms'),
        dcc.Input(id='bathrooms', type='number', value=2),
        html.Label('Floors'),
        dcc.Input(id='floors', type='number', value=1),
        html.Label('Square Feet Above'),
        dcc.Input(id='sqft-above', type='number', value=1000),
        html.Label('Square Feet Lot'),
        dcc.Input(id='sqft-lot', type='number', value=5000),
    ]),
    html.Button('Predict', id='predict-button', n_clicks=0),
    html.Div(id='prediction-output')
])

# Callback to update scatter plot
@app.callback(
    Output('scatter-plot', 'figure'),
    Input('total-sf', 'value')
)
def update_scatter_plot(total_sf):
    fig = px.scatter(df, x='TotalSF', y='price', trendline="ols", title="Price vs. Total SF")
    return fig

# Callback to update feature importance plot
@app.callback(
    Output('feature-importance', 'figure'),
    Input('total-sf', 'value')
)
def update_feature_importance(total_sf):
    importance = model.feature_importances_
    feature_names = ['TotalSF', 'sqft_living', 'bedrooms', 'bathrooms', 'floors', 'sqft_above', 'sqft_lot']
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
    fig = px.bar(feature_importance_df, x='Feature', y='Importance', title='Feature Importance')
    return fig

# Callback to make predictions
@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    Input('total-sf', 'value'),
    Input('sqft-living', 'value'),
    Input('bedrooms', 'value'),
    Input('bathrooms', 'value'),
    Input('floors', 'value'),
    Input('sqft-above', 'value'),
    Input('sqft-lot', 'value')
)
def predict(n_clicks, total_sf, sqft_living, bedrooms, bathrooms, floors, sqft_above, sqft_lot):
    if n_clicks > 0:
        features = pd.DataFrame([[total_sf, sqft_living, bedrooms, bathrooms, floors, sqft_above, sqft_lot]],
                                columns=['TotalSF', 'sqft_living', 'bedrooms', 'bathrooms', 'floors', 'sqft_above', 'sqft_lot'])
        prediction = model.predict(features)[0]
        return f'Predicted House Price: ${prediction:,.2f}'
    return ''

if __name__ == '__main__':
    app.run_server(debug=True)
 
