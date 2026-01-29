import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.correlation_tools import cov_nearest


def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads the Ken French 48 Industry Portfolios CSV.
    The 'Date' column is in YYYYMM format; we keep it as an integer index to make slicing easy.
    """
    df = pd.read_csv(file_path)
    if "Date" not in df.columns:
        raise ValueError("The CSV must contain a 'Date' column.")

    # Ensure YYYYMM integer index (avoids pandas parse-date warnings)
    df["Date"] = pd.to_numeric(df["Date"], errors="raise").astype(int)
    df = df.set_index("Date")
    return df


def extract_last_five_years(data: pd.DataFrame, industry):
    """
    Returns the last 5 years (monthly) of returns for the selected industry/industries.

    - If `industry` is a string -> returns a DataFrame with one column
    - If `industry` is a list of strings -> returns a DataFrame with those columns
    """
    end_key = int(data.index[-1])
    end_dt = pd.to_datetime(str(end_key), format="%Y%m")
    start_dt = end_dt - pd.DateOffset(years=5)
    start_key = int(f"{start_dt.year}{start_dt.month:02d}")

    if isinstance(industry, str):
        industry_cols = [industry]
    else:
        industry_cols = list(industry)

    return data.loc[start_key:end_key, industry_cols]


def compute_sigma(data: pd.DataFrame) -> pd.DataFrame:
    """
    Computes a covariance matrix and nudges it to the nearest PSD matrix (cov_nearest),
    which helps avoid numerical issues in matrix inversion.
    """
    if isinstance(data, pd.Series):
        data = data.to_frame()

    sigma = data.cov()
    sigma_psd = cov_nearest(sigma, method="nearest")
    return pd.DataFrame(sigma_psd, index=sigma.index, columns=sigma.columns)


def _ensure_returns_df(returns) -> pd.DataFrame:
    """
    Make the code robust if a single industry (Series) is passed.
    """
    if isinstance(returns, pd.Series):
        return returns.to_frame()
    if isinstance(returns, np.ndarray):
        if returns.ndim == 1:
            return pd.DataFrame(returns, columns=["Asset"])
        return pd.DataFrame(returns)
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame/Series or a numpy array.")
    return returns


def efficient_frontier_closed_form(
    returns: pd.DataFrame,
    sigma: pd.DataFrame,
    n_ptf: int = 100,
    annualize: bool = True
):
    """
    Closed-form efficient frontier for risky assets only.
    Returns:
        weights_df: (n_ptf x N) dataframe of weights
        variances: list of portfolio variances
        target_returns: array of target means
        mu: vector of asset expected returns (annualized if annualize=True)
    """
    returns = _ensure_returns_df(returns)
    time_factor = 12 if annualize else 1

    Sigma = sigma.values * time_factor
    inv_sigma = np.linalg.inv(Sigma)

    ones = np.ones((Sigma.shape[0], 1))
    mu = returns.mean().to_numpy().reshape(-1, 1) * time_factor

    A = (ones.T @ inv_sigma @ ones).item()
    B = (ones.T @ inv_sigma @ mu).item()
    C = (mu.T @ inv_sigma @ mu).item()
    D = A * C - B ** 2

    target_returns = np.linspace(mu.min(), mu.max() * 2, n_ptf)

    weights_list = []
    variances = []

    for r in target_returns:
        lam = (C - B * r) / D
        gam = (A * r - B) / D
        weights = inv_sigma @ (lam * ones + gam * mu)
        weights_list.append(weights.flatten())
        variances.append((A * r ** 2 - 2 * B * r + C) / D)

    weights_df = pd.DataFrame(weights_list, columns=returns.columns)
    return weights_df, variances, target_returns, mu.flatten()


def mean_variance_locus_with_rfr_notes(
    returns: pd.DataFrame,
    sigma: pd.DataFrame,
    n_ptf: int = 200,
    R: float = -5,
    annualize: bool = True
):
    returns = _ensure_returns_df(returns)
    time_factor = 12 if annualize else 1

    zbar = returns.mean().to_numpy().reshape(-1, 1) * time_factor
    Sigma = sigma.values * time_factor
    invSigma = np.linalg.inv(Sigma)

    n = Sigma.shape[0]
    ones = np.ones((n, 1))

    A = (ones.T @ invSigma @ ones).item()
    B = (ones.T @ invSigma @ zbar).item()
    C = (zbar.T @ invSigma @ zbar).item()

    denom = C - 2 * R * B + (R ** 2) * A

    zbar_min = float(zbar.min())
    zbar_max = float(zbar.max())
    target_returns = np.linspace(min(R, zbar_min * 2), max(R, zbar_max * 2), n_ptf)

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



def tangency_portfolio(
    returns: pd.DataFrame,
    sigma: pd.DataFrame,
    R: float,
    annualize: bool = True
):
    returns = _ensure_returns_df(returns)
    time_factor = 12 if annualize else 1

    zbar = returns.mean().to_numpy().reshape(-1, 1) * time_factor
    Sigma = sigma.values * time_factor
    invSigma = np.linalg.inv(Sigma)

    n = Sigma.shape[0]
    ones = np.ones((n, 1))

    A = (ones.T @ invSigma @ ones).item()
    B = (ones.T @ invSigma @ zbar).item()

    denom = (B - A * R)
    if abs(denom) < 1e-12:
        raise ValueError("B - A*R is too close to zero; tangency portfolio is ill-defined for this R.")

    w = (invSigma @ (zbar - R * ones)) / denom
    w = w.flatten()

    mu_tan = float(zbar.flatten() @ w)
    var_tan = float(w @ (Sigma @ w))
    sig_tan = float(np.sqrt(var_tan))
    sharpe_tan = (mu_tan - R) / sig_tan if sig_tan > 0 else np.nan

    w_series = pd.Series(w, index=returns.columns, name="w_tan")
    return w_series, mu_tan, var_tan, sig_tan, sharpe_tan


def verify_tangency_max_sharpe(
    target_returns: np.ndarray,
    variances: list,
    R: float
):

    sigmas = np.sqrt(np.array(variances, dtype=float))
    sharpes = (np.array(target_returns, dtype=float) - R) / sigmas
    idx_max = int(np.nanargmax(sharpes))
    return float(sharpes[idx_max]), idx_max

def main():
    data = load_data('data/48_Industry_Portfolios.csv')

    industries = ['Books', 'Soda', 'FabPr', 'Steel', 'Aero']
    rets = extract_last_five_years(data, industries)

    sigma = compute_sigma(rets)

    weights_df, variances, target_returns, mu = efficient_frontier_closed_form(
        rets, sigma, n_ptf=500, annualize=True
    )

    R = 2
    rf_res = mean_variance_locus_with_rfr_notes(
        rets, sigma, n_ptf=250, R=R, annualize=True
    )


    # Portefeuille tengent
    w_tan, mu_tan, var_tan, sig_tan, sharpe_tan = tangency_portfolio(rets, sigma, R=R, annualize=True)

    sep = "=" * 72
    print(f"\n{sep}")
    print("QUESTION 3 — Portefeuille tangent (risky-only) et ratio de Sharpe")
    print(sep)
    print(f"Taux sans risque R (annuel, en %): {R:.6f}")
    print(f"Somme des poids (doit être 1): {w_tan.sum():.10f}")
    print(f"Moyenne du portefeuille tangent μ_tan (%/an): {mu_tan:.6f}")
    print(f"Variance du portefeuille tangent Var_tan: {var_tan:.6f}")
    print(f"Écart-type du portefeuille tangent σ_tan (%/an): {sig_tan:.6f}")
    print(f"Sharpe_tan = (μ_tan - R)/σ_tan: {sharpe_tan:.6f}")

    print("\nPoids par industrie (w_tan):")
    print(w_tan.to_string(float_format=lambda x: f"{x:+.6f}"))

    sharpe_max, idx_max = verify_tangency_max_sharpe(target_returns, variances, R=R)
    print("\nVérification numérique (frontière efficiente risquée):")
    print(f"Sharpe maximum sur la frontière (discrétisée): {sharpe_max:.6f}")
    print(f"Différence |Sharpe_max - Sharpe_tan|: {abs(sharpe_max - sharpe_tan):.6e}")

    w_frontier_max = weights_df.iloc[idx_max]
    w_frontier_max = w_frontier_max / w_frontier_max.sum()
    print("\nPoids du portefeuille (frontière) au Sharpe max:")
    print(w_frontier_max.to_string(float_format=lambda x: f"{x:+.6f}"))
    print(sep)

    asset_sigmas = np.sqrt(np.diag(sigma.values * 12))

    plt.figure(figsize=(9, 6))
    plt.scatter(asset_sigmas, mu, marker='o')
    for i, ind in enumerate(industries):
        plt.text(asset_sigmas[i], mu[i], f" {ind}", va='center')

    plt.plot(np.sqrt(variances), target_returns, color='red', linewidth=2, label='Sans actif sans risque')

    # Usually only the upper ray is shown as "efficient" with risk-free (CML),
    # but we keep both (as in your original code) to match the expected format.
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
