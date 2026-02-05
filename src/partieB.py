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



from scipy.special import beta as beta_fn, betainc


def estimate_theta_adj(returns: pd.DataFrame, rf: float = 0.0) -> float:
    """
    Adjusted estimator of the squared maximum Sharpe ratio (theta_adj),
    following the Kan & Zhou adjustment used in MAXSER (Ao et al.).
    """
    R = returns - rf
    T = R.shape[0]
    N = R.shape[1]
    if T <= N + 2:
        raise ValueError(f"Need T > N + 2 for theta_adj; got T={T}, N={N}.")

    mu = R.mean().values.reshape(-1, 1)   # N x 1
    Sigma = R.cov().values               # N x N

    # Plug-in estimator: theta_s = mu' Sigma^{-1} mu
    try:
        x = np.linalg.solve(Sigma, mu)
    except np.linalg.LinAlgError:
        x = np.linalg.pinv(Sigma) @ mu
    theta_s = float((mu.T @ x).item())

    theta_s = max(theta_s, 0.0)
    if theta_s == 0.0:
        return 0.0

    # Adjustment term based on incomplete beta
    a = N / 2.0
    b = (T - N) / 2.0
    u = theta_s / (1.0 + theta_s)

    B_u = beta_fn(a, b) * betainc(a, b, u)

    if B_u <= 0:
        adj_term = 0.0
    else:
        adj_term = (2.0 * (theta_s ** (N / 2.0)) * ((1.0 + theta_s) ** (-(T - 2.0) / 2.0))) / (T * B_u)

    theta_adj = ((T - N - 2.0) / T) * theta_s - (N / T) + adj_term
    return float(theta_adj)


def compute_rc(theta_adj: float, sigma_target: float) -> float:
    """
    r^c = ((1 + theta) / sqrt(theta)) * sigma
    """
    if theta_adj <= 0:
        raise ValueError(f"theta_adj must be > 0 to compute r^c; got {theta_adj}.")
    return ((1.0 + theta_adj) / np.sqrt(theta_adj)) * sigma_target


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
    Solve MAXSER feasible LASSO:
        min_w (1/T) * sum_t (rc - w'R_t)^2
        s.t. ||w||_1 <= l1_bound
    Return w (N,)
    """
    X = R.values  # T x N
    T, N = X.shape

    w = cp.Variable(N)
    resid = rc - X @ w
    objective = cp.Minimize((1.0 / T) * cp.sum_squares(resid))
    constraints = [cp.norm1(w) <= float(l1_bound)]
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
            fold_mse.append(float(np.mean(resid_val ** 2)))

        mse_grid.append(float(np.mean(fold_mse)))

    best_idx = int(np.argmin(mse_grid))
    return float(l1_grid[best_idx])


def scale_to_target_vol(w: pd.Series, Sigma: pd.DataFrame, sigma_target: float) -> pd.Series:
    """
    Scale weights so that sqrt(w' Sigma w) == sigma_target
    """
    wv = w.values.reshape(-1, 1)
    vol = float(np.sqrt((wv.T @ Sigma.values @ wv).item()))
    if vol <= 0:
        raise ValueError("Computed portfolio vol is non-positive.")
    return w * (sigma_target / vol)


def run_maxser_from_returns(
    industry_returns: pd.DataFrame,
    rf: float = 0.0,
    sigma_target_annual: float = 0.10,
    l1_min: float = 0.05,
    l1_max: float = 5.0,
    l1_grid_size: int = 25,
    k_folds: int = 10
):
    """
    Run MAXSER using an already-loaded returns DataFrame (T x N).
    Does NOT load data. Designed for notebook integration.

    Returns:
        w_maxser: pd.Series (N,)
        diagnostics: dict
    """
    # theta_adj
    theta_adj = estimate_theta_adj(industry_returns, rf=rf)

    # rc from target sigma
    sigma_target_monthly = sigma_target_annual / np.sqrt(12)
    rc = compute_rc(theta_adj, sigma_target_monthly)

    # CV grid for L1 bound
    l1_grid = np.linspace(l1_min, l1_max, int(l1_grid_size))
    best_l1 = cv_select_l1_bound(industry_returns, rc=rc, l1_grid=l1_grid, k=int(k_folds))

    # LASSO
    w_lasso = solve_maxser_l1(industry_returns, rc=rc, l1_bound=best_l1)
    w_lasso = pd.Series(w_lasso, index=industry_returns.columns)

    # Scale to target vol
    Sigma_10y = industry_returns.cov()
    w_maxser = scale_to_target_vol(w_lasso, Sigma_10y, sigma_target_monthly)

    # Performance
    ptf_ret = industry_returns @ w_maxser
    mu_m = ptf_ret.mean()
    vol_m = ptf_ret.std()

    mu_ann = mu_m * 12
    vol_ann = vol_m * np.sqrt(12)
    sharpe_ann = (mu_ann / vol_ann) if vol_ann > 0 else np.nan

    diagnostics = {
        "theta_adj": float(theta_adj),
        "rc": float(rc),
        "best_l1": float(best_l1),
        "nonzero": int((w_lasso.abs() > 1e-6).sum()),
        "scaled_vol_monthly": float(np.sqrt((w_maxser.values @ Sigma_10y.values @ w_maxser.values).item())),
        "mu_ann": float(mu_ann),
        "vol_ann": float(vol_ann),
        "sharpe_ann": float(sharpe_ann),
    }
    return w_maxser, diagnostics


def main():
    # Load data only when running this file directly (NOT when importing)
    industry_returns = pd.read_csv('data/48_Industry_Portfolios.csv', na_values=-99.99)
    industry_returns['Date'] = pd.to_datetime(industry_returns['Date'], format='%Y%m')
    industry_returns.set_index('Date', inplace=True)
    industry_returns = industry_returns.sort_index()
    industry_returns = industry_returns.iloc[-120:]
    industry_returns = industry_returns / 100.0
    industry_returns = industry_returns.dropna()

    w_maxser, diag = run_maxser_from_returns(industry_returns, rf=0.0, sigma_target_annual=0.10)

    print("theta_adj =", diag["theta_adj"])
    print("rc =", diag["rc"])
    print("best_l1 =", diag["best_l1"])
    print("Non-zero weights (abs>1e-6):", diag["nonzero"])
    print("Scaled vol (monthly):", diag["scaled_vol_monthly"])
    print("\nMAXSER in-sample performance (rf=0):")
    print("Mean (ann) =", diag["mu_ann"])
    print("Vol  (ann) =", diag["vol_ann"])
    print("Sharpe     =", diag["sharpe_ann"])

    # Show top weights (optional)
    print("\nTop 10 |weights|:")
    print(w_maxser.reindex(w_maxser.abs().sort_values(ascending=False).head(10).index))


if __name__ == "__main__":
    main()