import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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


def efficient_frontier_closed_form(returns: pd.DataFrame, sigma: pd.DataFrame, n_ptf = 100, annualize=True):
    time_factor = 12 if annualize else 1
    inv_sigma = np.linalg.inv(sigma.values*time_factor)
    ones = np.ones((len(sigma), 1))
    mu = returns.mean().values.reshape(-1, 1)*time_factor
    A = ones.T @ inv_sigma @ ones
    B = ones.T @ inv_sigma @ mu
    C = mu.T @ inv_sigma @ mu
    D = A * C - B ** 2

    target_returns = np.linspace(mu.min(), mu.max()*2, n_ptf)
    weights_list = []
    variances = []

    for r in target_returns:
        lambda_ = (C - B * r) / D
        gamma_ = (A * r - B) / D
        weights = inv_sigma @ (lambda_ * ones + gamma_ * mu)
        weights_list.append(weights.flatten())
        variances.append(
            ((A*r**2-2*B*r+C) / D)[0] 
        )

    weights_df = pd.DataFrame(weights_list, columns=returns.columns)
    return weights_df, variances,target_returns,mu.flatten()

def main():
    data = load_data('data/48_Industry_Portfolios.csv')
    print(data.head())
    industries = ['Food', 'Soda', 'Beer', 'Smoke', 'Fin']
    data_last_five_years = extract_last_five_years(data, industries)

    print(data_last_five_years.head())
    sigma = compute_sigma(data_last_five_years)
    print(sigma)
    weights,variances,returns,mu = efficient_frontier_closed_form(data_last_five_years,sigma,annualize=True)
    
    for i, industry in enumerate(industries):
        plt.plot(np.sqrt(np.diag(sigma*12)), mu, 'o', label=industry)
        plt.text(np.sqrt(np.diag(sigma))[i]*np.sqrt(12), mu[i], industry)

    plt.plot(np.sqrt(variances), returns, color='red', linewidth=2)
    plt.xlabel('Risque (écart-type)')
    plt.ylabel('Retour attendu')
    plt.title('Frontière efficiente')
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()