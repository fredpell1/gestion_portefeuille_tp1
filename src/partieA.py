import pandas as pd

def load_data(file_path):
    return pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')

def extract_last_five_years(data, industry):
    end_date = data.index[-1]
    start_date = pd.to_datetime(end_date, format='%Y%m') - pd.DateOffset(years=5)
    start_date = pd.to_numeric(f"{start_date.year}{start_date.month:02d}")
    return data.loc[start_date:end_date, industry]


if __name__ == "__main__":
    data = load_data('data/48_Industry_Portfolios.csv')
    print(data.head())
    print(extract_last_five_years(data, ['Food', 'Soda', 'Beer', 'Smoke', 'Toys', 'Fun']))