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