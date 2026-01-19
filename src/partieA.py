import pandas as pd

def load_data(file_path):
    return pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')


if __name__ == "__main__":
    data = load_data('data/48_Industry_Portfolios.csv')
    print(data.head())