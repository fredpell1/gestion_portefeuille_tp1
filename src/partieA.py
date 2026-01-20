import pandas as pd
from statsmodels.stats.correlation_tools import cov_nearest



def load_data(file_path):
    return pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')

def extract_last_five_years(data, industry):
    end_date = data.index[-1]
    start_date = pd.to_datetime(end_date, format='%Y%m') - pd.DateOffset(years=5)
    start_date = pd.to_numeric(f"{start_date.year}{start_date.month:02d}")
    return data.loc[start_date:end_date, industry]

def compute_sigma(data: pd.DataFrame):
    sigma = data.cov()
    sigma = cov_nearest(sigma, method='nearest') # Assure la convergence de l'algorithme d'optimisation
    sigma_df = pd.DataFrame(sigma, index=data.columns, columns=data.columns)
    return sigma_df

def main():
    data = load_data('data/48_Industry_Portfolios.csv')
    print(data.head())
    industries = ['Food', 'Soda', 'Beer', 'Smoke', 'Toys', 'Fun']
    data_last_five_years = extract_last_five_years(data, industries)
    print(data_last_five_years.head())
    sigma = compute_sigma(data_last_five_years)
    print(sigma)


if __name__ == "__main__":
    main()