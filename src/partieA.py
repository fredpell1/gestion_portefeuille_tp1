import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.correlation_tools import cov_nearest
from scipy.optimize import minimize


def load_data(file_path):
    df = pd.read_csv(file_path)
    df["Date"] = pd.to_datetime(df["Date"].astype(str), format="%Y%m")
    df = df.set_index("Date")
    df = df.replace(-99.99, np.nan).dropna()  # important for Soda early history
    return df

def extract_last_five_years(data, industries):
    end_date = data.index.max()
    start_date = end_date - pd.DateOffset(years=5)
    return data.loc[start_date:end_date, industries]

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
            ((A*r**2-2*B*r+C) / D)[0][0]
        )

    weights_df = pd.DataFrame(weights_list, columns=returns.columns)
    return weights_df, variances,target_returns,mu.flatten()


def mean_variance_locus_with_rfr_notes(returns: pd.DataFrame, sigma: pd.DataFrame,
                                      n_ptf=200, R=2, annualize=True):
    time_factor = 12 if annualize else 1

    # annualized inputs
    zbar = returns.mean().values.reshape(-1, 1) * time_factor
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


def efficient_frontier_numerical(returns: pd.DataFrame, sigma: pd.DataFrame, n_ptf = 100, annualize=True,short_selling=False):
    """
    min 1/2 w.T @ sigma @ w
    s.t
        mu.T @ w = r
        ones.T @ w = 1
        w >= 0
    
    """
    time_factor = 12 if annualize else 1
    sigma_t = sigma.values*time_factor
    mu = returns.mean().values.reshape(-1, 1)*time_factor
    mu_assets = mu.flatten()

    target_returns = np.linspace(mu_assets.min(), mu_assets.max(), n_ptf)
    def obj_fn(w):
        return 0.5 * w.T @ sigma_t @ w
    weights_list = []
    variances = []
    for r in target_returns:
        cons = (
            {'type': 'eq', 'fun': lambda w: mu.T @ w - r},
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        )
        if not short_selling:
            bounds = [(0, None) for _ in range(len(sigma))]
        else:
            bounds = [(-np.inf, np.inf) for _ in range(len(sigma))]
        w0 = np.ones(len(sigma)) / len(sigma)
        res = minimize(obj_fn, w0, constraints=cons, bounds=bounds)
        if not res.success:
            pass
        weights_list.append(res.x)
        variances.append(2*res.fun)

    weights_df = pd.DataFrame(weights_list, columns=returns.columns)
    return weights_df, variances,target_returns,mu.flatten()

def efficient_frontier_with_rfr_noshort(returns, sigma, R, n_ptf=100, annualize=True):
    time_factor = 12 if annualize else 1
    sigma_t = sigma.values * time_factor
    mu = returns.mean().values.reshape(-1, 1) * time_factor
    mu_assets = mu.flatten()

    target_returns = np.linspace(mu_assets.min(), mu_assets.max(), n_ptf)

    def obj_fn(w):
        return 0.5 * w.T @ sigma_t @ w

    weights_list = []
    variances = []

    for mu0 in target_returns:
        cons = (
            {'type': 'eq', 'fun': lambda w, mu0=mu0: (mu - R).T @ w - (mu0 - R)},
            {'type': 'ineq', 'fun': lambda w: 1 - np.sum(w)}
        )
        bounds = [(0, None) for _ in range(len(sigma))]
        w0 = np.ones(len(sigma)) / len(sigma)
        res = minimize(obj_fn, w0, method='SLSQP', constraints=cons, bounds=bounds)
        if not res.success:
            continue
        weights_list.append(res.x)
        variances.append(2 * res.fun)

    weights_df = pd.DataFrame(weights_list, columns=returns.columns)
    return weights_df, variances, target_returns, mu.flatten()

# =========================
# TP1 - Partie A - Q6
# Tangency portfolio WITH risk-free asset + NO short-selling (R = 2)
# maximize Sharpe(w) = (mu'w - R) / sqrt(w' Sigma w)
# s.t. sum(w)=1 and w_i >= 0
# =========================
def tangency_portfolio_noshort(mu, Sigma, R=2):
    n = len(mu)

    def neg_sharpe(w):
        port_return = float(mu @ w)
        port_variance = float(w @ Sigma @ w)
        if port_variance <= 1e-12:
            return 1e9  # penalty
        sharpe_ratio = (port_return - R) / np.sqrt(port_variance)
        return -sharpe_ratio

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1},)
    bounds = [(0, None) for _ in range(n)]
    w0 = np.ones(n) / n

    result = minimize(neg_sharpe, w0, method='SLSQP', bounds=bounds, constraints=constraints)

    if not result.success:
        raise ValueError(f"Optimization did not converge: {result.message}")

    w_star = result.x
    mu_p = float(mu @ w_star)
    var_p = float(w_star @ Sigma @ w_star)
    sharpe_p = (mu_p - R) / np.sqrt(var_p)

    return w_star, mu_p, var_p, sharpe_p
def verify_max_sharpe_random(mu, Sigma, R=2, n_random=20000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(mu)

    # Dirichlet => poids >=0 et somme=1
    W = rng.dirichlet(np.ones(n), size=n_random)

    port_returns = W @ mu
    port_vars = np.einsum("ij,jk,ik->i", W, Sigma, W)
    port_sigs = np.sqrt(np.maximum(port_vars, 1e-12))
    sharpes = (port_returns - R) / port_sigs

    i_best = int(np.argmax(sharpes))
    return float(sharpes[i_best]), W[i_best]

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



def verify_tangency_max_sharpe(
    target_returns: np.ndarray,
    variances: list,
    R: float
):

    sigmas = np.sqrt(np.array(variances, dtype=float))
    sharpes = (np.array(target_returns, dtype=float) - R) / sigmas
    idx_max = int(np.nanargmax(sharpes))
    return float(sharpes[idx_max]), idx_max

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

def main():
    data = load_data('data/48_Industry_Portfolios.csv')

    industries = ['Food', 'Soda', 'Beer', 'Smoke', 'Fin']
    data_last_five_years = extract_last_five_years(data, industries)

    sigma = compute_sigma(data_last_five_years)
    print(sigma)
    weights_cf,variances_cf,returns_cf,mu_cf = efficient_frontier_closed_form(data_last_five_years,sigma,annualize=True)
    weights_num,variances_num,returns_num,mu_num = efficient_frontier_numerical(data_last_five_years,sigma,annualize=True,short_selling=False)

    print(f"Comparing mu_cf vs mu_num: {np.allclose(mu_cf, mu_num)}")
    print(f"Comparing returns_cf vs returns_num: {np.allclose(returns_cf, returns_num)}")
    print(f"Comparing variances_cf vs variances_num: {np.allclose(variances_cf, variances_num)}")
    print(f"Comparing weights_cf vs weights_num: {np.allclose(weights_cf.values, weights_num.values, atol=1e-3, rtol=1e-3)}")


    # for i,w in enumerate(zip(weights_cf.values, weights_num.values)):
    #     w_cf, w_num = w
    #     print(f"Weight mismatch for index {i}: {w_cf} vs {w_num}")
    #     print(w_cf - w_num)


    # print(weights_cf)
    asset_sigmas = np.sqrt(np.diag(sigma.values * 12))
    for i, industry in enumerate(industries):
        plt.scatter(asset_sigmas[i], mu_cf[i])
        plt.text(asset_sigmas[i], mu_cf[i], industry)

    plt.plot(np.sqrt(variances_cf), returns_cf, color='red', linewidth=2, label='Frontière efficiente (Avec vente à découvert)')
    plt.xlabel('Risque (Écart-type annuel)')
    plt.ylabel('Rendement attendu annuel (%)')
    plt.title('Locus Moyenne-Variance : Analyse comparative')
    plt.legend()
    plt.grid(True)
    plt.show()

    weights, variances, rets, mu = efficient_frontier_closed_form(
        data_last_five_years, sigma, annualize=True
    )

    R = 2
    rf_res = mean_variance_locus_with_rfr_notes(
        data_last_five_years, sigma, n_ptf=250, R=R, annualize=True
    )

    plt.figure(figsize=(9, 6))
    plt.scatter(asset_sigmas, mu, marker='o', label='Actifs individuels')
    for i, ind in enumerate(industries):
        plt.text(asset_sigmas[i], mu[i], f" {ind}", va='center')

    plt.plot(np.sqrt(variances), rets, color='red', linewidth=2, label='Frontière sans actif sans risque')
    plt.plot(rf_res["sigmas_line"], rf_res["mu_plus"], linestyle='--', linewidth=2, label='Frontière avec actif sans risque (+)')
    plt.plot(rf_res["sigmas_line"], rf_res["mu_minus"], linestyle='--', linewidth=2, label='Frontière avec actif sans risque (-)')
    plt.scatter([0.0], [R], marker='x', s=100, color='black', label='Actif sans risque (R)')
    plt.text(0.0, R, " R", va='bottom')

    plt.xlabel('Risque (Écart-type annuel)')
    plt.ylabel('Rendement attendu annuel (%)')
    plt.title('Locus Moyenne-Variance : Sans / Avec actif sans risque')
    plt.grid(True)
    plt.legend()
    # print(np.sqrt(variances_cf))
    # print(returns_cf)

    # print(weights_num)
    # print(np.sqrt(variances_num))
    # print(returns_num)
    for i, industry in enumerate(industries):
        plt.scatter(asset_sigmas[i], mu_cf[i])
        plt.text(asset_sigmas[i], mu_cf[i], industry)

    plt.plot(np.sqrt(variances_num), returns_num, color='blue', linewidth=2,linestyle='--', label='Frontière efficiente (Sans vente à découvert)')
    plt.xlabel('Risque (Écart-type annuel)')
    plt.ylabel('Rendement attendu annuel (%)')
    plt.title('Locus Moyenne-Variance (Sans vente à découvert)')
    plt.legend()
    plt.grid(True)

    
    # Q5 - Mean-Variance Locus (with risk-free asset, no short-selling)
    R = 2
    weights_noshort, variances_noshort, rets_noshort, mu_noshort = efficient_frontier_with_rfr_noshort(
    data_last_five_years, sigma, R, n_ptf=250, annualize=True
)

    sig_q5 = np.sqrt(np.array(variances_noshort))
    mu_q5 = np.array(rets_noshort)[:len(sig_q5)]

    plt.plot(sig_q5, mu_q5, linestyle='-.', linewidth=2,
         label='Locus (Actif sans risque + sans vente à découvert)')
    plt.xlabel('Risque (Écart-type annuel)')
    plt.ylabel('Rendement attendu annuel (%)')
    plt.title('Locus Moyenne-Variance avec actif sans risque (Sans vente à découvert)')
    plt.legend()
    plt.grid(True)

    # Q6 - Tangency portfolio (Rf = 2, no short-selling)
    R = 2
    mu_assets = data_last_five_years.mean().values * 12
    Sigma_assets = sigma.values * 12

    w_star, mu_p, var_p, sharpe_p = tangency_portfolio_noshort(mu_assets, Sigma_assets, R)

    print("\n================ Q6: Tangency (Rf=0, no-short) ================")
    print("Poids du portefeuille tangent (long-only):")
    for name, w in zip(industries, w_star):
        print(f"  {name}: {w:.4f}")
    print(f"Moyenne (mu_p): {mu_p:.6f}")
    print(f"Variance (var_p): {var_p:.6f}")
    print(f"Ecart-type (sigma_p): {np.sqrt(var_p):.6f}")
    print(f"Sharpe (mu_p - R)/sigma_p: {sharpe_p:.6f}")

    max_sharpe_rand, w_best_rand = verify_max_sharpe_random(mu_assets, Sigma_assets, R, n_random=20000, seed=0)
    print(f"Max Sharpe (random long-only, n=20000): {max_sharpe_rand:.6f}")
    print(f"Diff (tangency - random max): {sharpe_p - max_sharpe_rand:.6e}")


    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()