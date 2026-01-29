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

data = load_data('data/48_Industry_Portfolios.csv')

def compute_sigma(data: pd.DataFrame):
    sigma = data.cov()
    sigma = cov_nearest(sigma, method='nearest')
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


def mean_variance_locus_with_rfr_notes(returns: pd.DataFrame, sigma: pd.DataFrame,
                                      n_ptf=200, R=-5, annualize=True):
    time_factor = 12 if annualize else 1

    # annualized inputs
    zbar = returns.mean().values.reshape(-1, 1) * time_factor
    Sigma = sigma.values * time_factor

    invSigma = np.linalg.inv(Sigma)
    n = Sigma.shape[0]
    ones = np.ones((n, 1))

    
    A = float(ones.T @ invSigma @ ones)
    B = float(ones.T @ invSigma @ zbar)
    C = float(zbar.T @ invSigma @ zbar)

    denom = C - 2 * R * B + (R ** 2) * A  

    
    zbar_min = float(zbar.min())
    zbar_max = float(zbar.max())
    target_returns = np.linspace(min(R, zbar_min*2), max(R, zbar_max*2), n_ptf)

    weights_list = []
    variances = []

    for mu0 in target_returns:
        gamma = (mu0 - R) / denom
        w = gamma * (invSigma @ (zbar - R * ones))  
        var = ((mu0 - R) ** 2) / denom              

        weights_list.append(w.flatten())
        variances.append(float(var))

    weights_df = pd.DataFrame(weights_list, columns=returns.columns)

    sigma_max = np.sqrt(max(variances)) if len(variances) else 0.0
    sigmas = np.linspace(0.0, sigma_max, n_ptf)
    slope = np.sqrt(denom)
    mu_plus = R + slope * sigmas
    mu_minus = R - slope * sigmas

    return {
        "weights_risky": weights_df,
        "variances": variances,
        "target_returns": target_returns,
        "A": A, "B": B, "C": C, "denom": denom,
        "sigmas_line": sigmas,
        "mu_plus": mu_plus,
        "mu_minus": mu_minus,
    }


def main():
    data = load_data('data/48_Industry_Portfolios.csv')

    industries = ['Books', 'Soda', 'FabPr', 'Steel', 'Aero']
    data_last_five_years = extract_last_five_years(data, industries)

    sigma = compute_sigma(data_last_five_years)

    weights, variances, rets, mu = efficient_frontier_closed_form(
        data_last_five_years, sigma, annualize=True
    )

    R = 2
    rf_res = mean_variance_locus_with_rfr_notes(
        data_last_five_years, sigma, n_ptf=250, R=R, annualize=True
    )

    asset_sigmas = np.sqrt(np.diag(sigma.values * 12))
    plt.figure(figsize=(9, 6))
    plt.scatter(asset_sigmas, mu, marker='o')
    for i, ind in enumerate(industries):
        plt.text(asset_sigmas[i], mu[i], f" {ind}", va='center')

    plt.plot(np.sqrt(variances), rets, color='red', linewidth=2, label='Sans actif sans risque')
    plt.plot(rf_res["sigmas_line"], rf_res["mu_plus"], linestyle='--', linewidth=2, label='Avec actif sans risque (+)')
    plt.plot(rf_res["sigmas_line"], rf_res["mu_minus"], linestyle='--', linewidth=2, label='Avec actif sans risque (-)')
    plt.scatter([0.0], [R], marker='x')
    plt.text(0.0, R, " R", va='bottom')

    plt.xlabel('Risque (écart-type)')
    plt.ylabel('Retour attendu')
    plt.title('Mean-Variance Locus: sans / avec actif sans risque')
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()