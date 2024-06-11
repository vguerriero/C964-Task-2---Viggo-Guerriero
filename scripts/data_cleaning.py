import pandas as pd

def clean_data(input_path, output_path):
    df = pd.read_csv(input_path)
    df = df.dropna()
    df = df.drop_duplicates()
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    clean_data('data/raw/house_prices.csv', 'data/cleaned/cleaned_data.csv')

