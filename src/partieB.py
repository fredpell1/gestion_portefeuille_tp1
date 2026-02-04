import numpy as np
import pandas as pd
from scipy.optimize import minimize
import cvxpy as cp

def efficient_frontier_asset_subset(returns: pd.DataFrame, sigma: pd.DataFrame, n_ptf = 100, annualize=True,short_selling=False,rf=None,max_asset = 5):
    """
    min 1/2 w.T @ sigma @ w \\
    s.t \\
        mu.T @ w = r \\
        ones.T @ w = 1 \\
        ones.T @ y = max_asset \\
        abs(w) <= y \\
        y in {0,1} \\
        w >= 0 
    
    """
    time_factor = 12 if annualize else 1
    sigma_t = sigma.values*time_factor
    mu = returns.mean().values.reshape(-1, 1)*time_factor
    mu_assets = mu.flatten()

    target_returns = np.linspace(mu_assets.min(), mu_assets.max(), n_ptf)

    weights_list = []
    variances = []
    for r in target_returns:
        y = cp.Variable(len(sigma), boolean=True)
        w = cp.Variable(len(sigma))
    
        constraints = [
            cp.sum(y) <= max_asset,
            cp.abs(w) <= y
        ]
    
        if not short_selling:
            constraints.append(w >= 0)
        if rf is not None:
            constraints += [
                (mu - rf).T @ w == r - rf
            ]
        else:
            constraints += [
                mu.T @ w == r,
                cp.sum(w) == 1
            ]
        if not short_selling:
            constraints.append(w >= 0)
        
        objective = cp.Minimize(0.5 * cp.quad_form(w, sigma_t))
        constraints = constraints
        prob = cp.Problem(objective, constraints)
        prob.solve(solver='SCIP')
        if prob.status != cp.OPTIMAL:
            pass
        weights_list.append(w.value)
        variances.append(prob.value*2)


    weights_df = pd.DataFrame(weights_list, columns=returns.columns)
    return weights_df, variances,target_returns,mu.flatten()


def tangency_portfolio_heuristic(risky_frontier,riskless_frontier,rf):
    idx = np.argmin(np.sqrt(risky_frontier['variance']) - np.sqrt(riskless_frontier['variance']))
    sharpe_ratio = (risky_frontier['return'][idx] - rf) / np.sqrt(risky_frontier['variance'][idx])
    weights_df = pd.DataFrame(risky_frontier['weights'],columns=risky_frontier['weights'].columns)
    return risky_frontier['weights'].iloc[idx], risky_frontier['variance'][idx], risky_frontier['return'][idx], sharpe_ratio


# Q4 — MAXSER allocation (Ao et al. 2018) on 48 industries
# Data: monthly returns, last 10 years (120 months)
# Empirical choice: rf = 0 (use raw returns as excess returns)


industry_returns = pd.read_csv('data/48_Industry_Portfolios.csv', na_values=-99.99)
industry_returns['Date'] = pd.to_datetime(industry_returns['Date'], format='%Y%m')
industry_returns.set_index('Date', inplace=True)
industry_returns = industry_returns.sort_index()
industry_returns = industry_returns.iloc[-120:]
industry_returns = industry_returns / 100.0
industry_returns = industry_returns.dropna()
industry_sigma = industry_returns.cov()
rf = 0.0

# Estimate adjusted maximum Sharpe ratio (theta_adj)
# Use sample mean and covariance of returns, with small-sample adjustment
from scipy.special import beta as beta_fn, betainc

def estimate_theta_adj(returns: pd.DataFrame, rf: float = 0.0) -> float:
    """
    Adjusted estimator of the squared maximum Sharpe ratio (theta_adj),
    as in the course slides (Kan & Zhou adjustment used in Ao et al. MAXSER).
    """
    R = returns - rf
    T = R.shape[0]
    N = R.shape[1]
    if T <= N + 2:
        raise ValueError(f"Need T > N + 2 for theta_adj; got T={T}, N={N}.")

    mu = R.mean().values.reshape(-1, 1)          
    Sigma = R.cov()
    Sigma = Sigma.values                        

    # estimator: theta_s = mu' Sigma^{-1} mu
    try:
        x = np.linalg.solve(Sigma, mu)
    except np.linalg.LinAlgError:
        x = np.linalg.pinv(Sigma) @ mu
    theta_s = float((mu.T @ x).item())

    # Numerical guard
    theta_s = max(theta_s, 0.0)
    if theta_s == 0.0:
        return 0.0
    

    # theta_adj = ((T-N-2)/T)*theta_s - N/T + adjustment_term
    a = N / 2.0
    b = (T - N) / 2.0
    u = theta_s / (1.0 + theta_s)  # argument of incomplete beta

    # Incomplete beta B_u(a,b) = Beta(a,b) * I_u(a,b)
    B_u = beta_fn(a, b) * betainc(a, b, u)

    # adjustment term 
    # 2 * (theta_s)^(N/2) * (1+theta_s)^(-(T-2)/2) / (T * B_u)
    if B_u <= 0:
        
        adj_term = 0.0
    else:
        adj_term = (2.0 * (theta_s ** (N / 2.0)) * ((1.0 + theta_s) ** (-(T - 2.0) / 2.0))) / (T * B_u)

    theta_adj = ((T - N - 2.0) / T) * theta_s - (N / T) + adj_term
    return float(theta_adj)

theta_adj = estimate_theta_adj(industry_returns, rf=0.0)
print("theta_adj =", theta_adj)

def compute_rc(theta_adj: float, sigma_target: float) -> float:
    """
    r^c = ((1 + theta) / sqrt(theta)) * sigma
    """
    if theta_adj <= 0:
        raise ValueError(f"theta_adj must be > 0 to compute r^c; got {theta_adj}.")
    return ((1.0 + theta_adj) / np.sqrt(theta_adj)) * sigma_target


# target annual volatility
sigma_target_annual = 0.10
sigma_target_monthly = sigma_target_annual / np.sqrt(12)

rc = compute_rc(theta_adj, sigma_target_monthly)
print("rc =", rc)

def blocked_folds(T: int, k: int = 10):
    """
    Split indices into k contiguous folds (no shuffling): time-series-friendly CV.
    """
    fold_sizes = [(T // k) + (1 if i < (T % k) else 0) for i in range(k)]
    idx = np.arange(T)
    folds = []
    start = 0
    for fs in fold_sizes:
        end = start + fs
        folds.append(idx[start:end])
        start = end
    return folds


def solve_maxser_l1(R: pd.DataFrame, rc: float, l1_bound: float) -> np.ndarray:
    """
    Solve MAXSER LASSO:
        min_w (1/T) * sum_t (rc - w'R_t)^2
        s.t. ||w||_1 <= l1_bound
    Return w (N,)
    """
    X = R.values  # T x N
    T, N = X.shape

    w = cp.Variable(N)
    resid = rc - X @ w
    objective = cp.Minimize((1.0 / T) * cp.sum_squares(resid))
    constraints = [cp.norm1(w) <= l1_bound]

    prob = cp.Problem(objective, constraints)

    # Prefer OSQP/ECOS; fallback to SCS
    try:
        prob.solve(solver="OSQP", verbose=False)
    except Exception:
        try:
            prob.solve(solver="ECOS", verbose=False)
        except Exception:
            prob.solve(solver="SCS", verbose=False)

    if w.value is None:
        raise RuntimeError(f"MAXSER LASSO did not solve. Status: {prob.status}")

    return np.array(w.value).reshape(-1)


def cv_select_l1_bound(R: pd.DataFrame, rc: float, l1_grid: np.ndarray, k: int = 10) -> float:
    """
    Select l1_bound (lambda) by blocked CV: minimize average validation MSE.
    """
    X = R.values
    T, N = X.shape
    folds = blocked_folds(T, k=k)

    mse_grid = []

    for l1 in l1_grid:
        fold_mse = []
        for f in range(k):
            val_idx = folds[f]
            train_idx = np.setdiff1d(np.arange(T), val_idx)

            R_train = pd.DataFrame(X[train_idx, :], columns=R.columns)
            R_val = X[val_idx, :]

            w_hat = solve_maxser_l1(R_train, rc=rc, l1_bound=float(l1))
            resid_val = rc - R_val @ w_hat
            fold_mse.append(np.mean(resid_val ** 2))

        mse_grid.append(np.mean(fold_mse))

    best_idx = int(np.argmin(mse_grid))
    return float(l1_grid[best_idx])


# ---- L1 grid (empirical choice)
# Upper bound: large enough to be effectively unpenalized
# Lower bound: very sparse solution
N = industry_returns.shape[1]

l1_min = 0.05
l1_max = 5.0
l1_grid = np.linspace(l1_min, l1_max, 25)

best_l1 = cv_select_l1_bound(industry_returns, rc=rc, l1_grid=l1_grid, k=10)
print("best_l1 =", best_l1)

w_lasso = solve_maxser_l1(industry_returns, rc=rc, l1_bound=best_l1)
w_lasso = pd.Series(w_lasso, index=industry_returns.columns)

print("Non-zero weights (abs>1e-6):", int((w_lasso.abs() > 1e-6).sum()))
print("Top 10 |weights|:")
print(w_lasso.reindex(w_lasso.abs().sort_values(ascending=False).head(10).index))

def scale_to_target_vol(w: pd.Series, Sigma: pd.DataFrame, sigma_target: float) -> pd.Series:
    wv = w.values.reshape(-1, 1)
    vol = float(np.sqrt(wv.T @ Sigma.values @ wv))
    if vol <= 0:
        raise ValueError("Computed portfolio vol is non-positive.")
    return w * (sigma_target / vol)

# Use the sample covariance on the same 10y window
Sigma_10y = industry_returns.cov()
w_maxser = scale_to_target_vol(w_lasso, Sigma_10y, sigma_target_monthly)

print("Scaled vol (monthly):", float(np.sqrt(w_maxser.values @ Sigma_10y.values @ w_maxser.values)))

ptf_ret = industry_returns @ w_maxser
mu_m = ptf_ret.mean()
vol_m = ptf_ret.std()
sharpe_m = mu_m / vol_m if vol_m > 0 else np.nan

mu_ann = mu_m * 12
vol_ann = vol_m * np.sqrt(12)
sharpe_ann = mu_ann / vol_ann if vol_ann > 0 else np.nan

print("\nMAXSER in-sample performance (rf=0):")
print("Mean (ann) =", mu_ann)
print("Vol  (ann) =", vol_ann)
print("Sharpe     =", sharpe_ann)


