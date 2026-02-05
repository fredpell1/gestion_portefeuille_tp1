import numpy as np
import pandas as pd
from scipy.optimize import minimize
import cvxpy as cp
import matplotlib.pyplot as plt
from partieA import (
    compute_sigma,
    efficient_frontier_closed_form,
    efficient_frontier_numerical,
    tangency_portfolio,
    tangency_portfolio_noshort,
)

# ============================================================
# PARTIE B - Q1 Bootstrap (comparaisons avec Partie A)
# ============================================================

def b1_bootstrap_1000(
    returns_5: pd.DataFrame,
    industries: list,
    B: int = 1000,
    annualize: bool = True,
    R: float = 2,
    n_ptf: int = 80,
    seed: int = 0
):
    """
    Bootstrap sur la fenêtre (returns_5) et comparaison à l’échantillon original (Partie A).
    On conserve:
      - Q1: point médian de la frontière (short-selling autorisé)
      - Q4: point médian de la frontière (no-short)
      - Q3: portefeuille tangent (non-contraint) avec R
      - Q6: portefeuille tangent (no-short) avec R
    """
    rng = np.random.default_rng(seed)
    T = len(returns_5)
    n = len(industries)
    tf = 12 if annualize else 1

    # --------- Stockage bootstrap ----------
    # Q1 / Q4 : point médian (sigma, mu)
    ret_q1_mid = np.zeros(B); sig_q1_mid = np.zeros(B)
    ret_q4_mid = np.zeros(B); sig_q4_mid = np.zeros(B)

    # Q3 : tangency non-contraint
    mu_q3 = np.zeros(B); sig_q3 = np.zeros(B); var_q3 = np.zeros(B); sharpe_q3 = np.zeros(B)
    w_q3 = np.zeros((B, n))

    # Q6 : tangency no-short
    mu_q6 = np.zeros(B); sig_q6 = np.zeros(B); var_q6 = np.zeros(B); sharpe_q6 = np.zeros(B)
    w_q6 = np.zeros((B, n))

    for b in range(B):
        idx = rng.integers(0, T, size=T)
        boot = returns_5.iloc[idx].copy()
        boot.index = range(len(boot))

        sigma_b = compute_sigma(boot)

        # ----- Q1 (short-selling autorisé) : frontière sans Rf -----
        _, var1, rets1, _ = efficient_frontier_closed_form(
            boot, sigma_b, n_ptf=n_ptf, annualize=annualize
        )
        sig1 = np.sqrt(np.array(var1, dtype=float))
        mid1 = len(sig1) // 2
        sig_q1_mid[b] = float(sig1[mid1])
        ret_q1_mid[b] = float(rets1[mid1])

        # ----- Q4 (no-short) : frontière sans Rf -----
        _, var4, rets4, _ = efficient_frontier_numerical(
            boot, sigma_b, n_ptf=n_ptf, annualize=annualize, short_selling=False
        )
        sig4 = np.sqrt(np.array(var4, dtype=float))
        mid4 = len(sig4) // 2
        sig_q4_mid[b] = float(sig4[mid4])
        ret_q4_mid[b] = float(rets4[mid4])

        # ----- Q3 (tangency non-contraint) : avec R -----
        try:
            w_tan_b, mu_t, var_t, sig_t, sharpe_t = tangency_portfolio(
                boot, sigma_b, R=R, annualize=annualize
            )

            w_vec = np.array(w_tan_b.values, dtype=float)
            lev = float(np.sum(np.abs(w_vec)))          # mesure de levier / instabilité

            # Filtres simples (à ajuster si besoin)
            if (not np.isfinite(sig_t)) or (sig_t <= 0):
                raise ValueError("sigma non-finie")
            if (not np.isfinite(mu_t)) or (not np.isfinite(sharpe_t)):
                raise ValueError("mu/sharpe non-finis")

            # bornes “raisonnables” pour éviter les explosions
            if sig_t > 300 or lev > 50:                 # <- clés: sigma et leverage
                raise ValueError(f"instable: sigma={sig_t:.2f}, lev={lev:.2f}")

            mu_q3[b] = float(mu_t)
            sig_q3[b] = float(sig_t)
            var_q3[b] = float(var_t)
            sharpe_q3[b] = float(sharpe_t)
            w_q3[b, :] = w_vec

        except Exception:
            # on met NaN => on pourra filtrer au moment des stats/plots
            mu_q3[b] = np.nan
            sig_q3[b] = np.nan
            var_q3[b] = np.nan
            sharpe_q3[b] = np.nan
            w_q3[b, :] = np.nan

        # ----- Q6 (tangency no-short) : avec R -----
        mu_vec = boot.mean().values * tf
        Sigma_mat = sigma_b.values * tf
        w_star, mu_p, var_p, sharpe_p = tangency_portfolio_noshort(mu_vec, Sigma_mat, R=R)

        mu_q6[b] = float(mu_p)
        sig_q6[b] = float(np.sqrt(var_p))
        var_q6[b] = float(var_p)
        sharpe_q6[b] = float(sharpe_p)
        w_q6[b, :] = np.array(w_star, dtype=float)

    # --------- Statistiques bootstrap ----------
    def stats(x):
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if len(x) == 0:
            return {"moyenne": np.nan, "ecart_type": np.nan, "p05": np.nan, "p50": np.nan, "p95": np.nan}
        return {
            "moyenne": float(np.mean(x)),
            "ecart_type": float(np.std(x, ddof=0)),
            "p05": float(np.quantile(x, 0.05)),
            "p50": float(np.quantile(x, 0.50)),
            "p95": float(np.quantile(x, 0.95)),
        }

    # ============================================================
    # Valeurs "Partie A" = échantillon original (sans bootstrap)
    # ============================================================
    sigma_A = compute_sigma(returns_5)

    # Q1 (A) : frontière short-selling autorisé + point médian
    _, var1A, rets1A, _ = efficient_frontier_closed_form(
        returns_5, sigma_A, n_ptf=n_ptf, annualize=annualize
    )
    sig1A = np.sqrt(np.array(var1A, dtype=float))
    mid1A = len(sig1A) // 2
    q1_A_mid = {"sigma": float(sig1A[mid1A]), "mu": float(rets1A[mid1A])}

    # Q4 (A) : frontière no-short + point médian
    _, var4A, rets4A, _ = efficient_frontier_numerical(
        returns_5, sigma_A, n_ptf=n_ptf, annualize=annualize, short_selling=False
    )
    sig4A = np.sqrt(np.array(var4A, dtype=float))
    mid4A = len(sig4A) // 2
    q4_A_mid = {"sigma": float(sig4A[mid4A]), "mu": float(rets4A[mid4A])}

    # Q3 (A) : tangency non-contraint
    w_tan_A, mu_tan_A, var_tan_A, sig_tan_A, sharpe_tan_A = tangency_portfolio(
        returns_5, sigma_A, R=R, annualize=annualize
    )
    q3_A = {
        "w": w_tan_A,  # pandas Series (avec index industries)
        "mu": float(mu_tan_A),
        "var": float(var_tan_A),
        "sigma": float(sig_tan_A),
        "sharpe": float(sharpe_tan_A),
    }

    # Q6 (A) : tangency no-short
    mu_vec_A = returns_5.mean().values * tf
    Sigma_mat_A = sigma_A.values * tf
    w_star_A, mu_p_A, var_p_A, sharpe_p_A = tangency_portfolio_noshort(mu_vec_A, Sigma_mat_A, R=R)
    q6_A = {
        "w": pd.Series(w_star_A, index=industries),
        "mu": float(mu_p_A),
        "var": float(var_p_A),
        "sigma": float(np.sqrt(var_p_A)),
        "sharpe": float(sharpe_p_A),
    }

    return {
        "B": B,
        "R": R,
        "annualize": annualize,
        "n_ptf": n_ptf,
        "industries": industries,

        # Série bootstrap (pour plots)
        "series": {
            "sig_q1_mid": sig_q1_mid,
            "ret_q1_mid": ret_q1_mid,
            "sig_q4_mid": sig_q4_mid,
            "ret_q4_mid": ret_q4_mid,

            "mu_q3": mu_q3,
            "sig_q3": sig_q3,
            "var_q3": var_q3,
            "sharpe_q3": sharpe_q3,
            "w_q3": w_q3,

            "mu_q6": mu_q6,
            "sig_q6": sig_q6,
            "var_q6": var_q6,
            "sharpe_q6": sharpe_q6,
            "w_q6": w_q6,
        },

        # Résumés bootstrap
        "resume": {
            "Q1_point_median": {"sigma": stats(sig_q1_mid), "retour": stats(ret_q1_mid)},
            "Q4_point_median": {"sigma": stats(sig_q4_mid), "retour": stats(ret_q4_mid)},

            "Q3_tangency_non_contraint": {
                "mu": stats(mu_q3),
                "var": stats(var_q3),
                "sigma": stats(sig_q3),
                "sharpe": stats(sharpe_q3),
                "poids": {ind: stats(w_q3[:, j]) for j, ind in enumerate(industries)},
            },
            "Q6_tangency_no_short": {
                "mu": stats(mu_q6),
                "var": stats(var_q6),
                "sigma": stats(sig_q6),
                "sharpe": stats(sharpe_q6),
                "poids": {ind: stats(w_q6[:, j]) for j, ind in enumerate(industries)},
            },
        },

        # Valeurs Partie A (échantillon original)
        "A": {
            "Q1_mid": q1_A_mid,
            "Q4_mid": q4_A_mid,
            "Q3": q3_A,
            "Q6": q6_A,
            "frontieres": {
                "Q1_sig": sig1A,
                "Q1_mu": np.array(rets1A, dtype=float),
                "Q4_sig": sig4A,
                "Q4_mu": np.array(rets4A, dtype=float),
            }
        }
    }


def b1_print_sections_fr(res_b1: dict):
    """
    Impression FR, avec comparaison A (échantillon original) vs bootstrap.
    On SUPPRIME le print de Q2 (comme demandé).
    """
    sep = "=" * 86
    r = res_b1["resume"]
    A = res_b1["A"]
    inds = res_b1["industries"]
    R = res_b1["R"]

    def largeur_90(st):
        return st["p95"] - st["p05"]

    print("\n" + sep)
    print("PARTIE B — QUESTION 1 : BOOTSTRAP (incertitude d'estimation)")
    print(sep)
    print(f"Nombre de réplications bootstrap : {res_b1['B']}")
    print("Industries :", ", ".join(inds))
    print(f"Taux sans risque utilisé (unique) : R = {R}")

    # ---------------- Q1 ----------------
    print("\n" + "-" * 86)
    print("Q1 — Frontière efficiente (sans actif sans risque, short-selling autorisé)")
    s = r["Q1_point_median"]
    muA = A["Q1_mid"]["mu"]; sigA = A["Q1_mid"]["sigma"]

    print("Point médian de frontière (index ~50%) — Bootstrap vs Partie A :")
    print(f"  σ_boot (p50) = {s['sigma']['p50']:.4f} | 5%-95% = [{s['sigma']['p05']:.4f}, {s['sigma']['p95']:.4f}]"
          f" | σ_A = {sigA:.4f}")
    print(f"  μ_boot (p50) = {s['retour']['p50']:.4f} | 5%-95% = [{s['retour']['p05']:.4f}, {s['retour']['p95']:.4f}]"
          f" | μ_A = {muA:.4f}")

    print("Incertitude (bootstrap) :")
    print(f"  Largeur bande 90% sur σ : {largeur_90(s['sigma']):.4f}")
    print(f"  Largeur bande 90% sur μ : {largeur_90(s['retour']):.4f}")
    print(f"  Écart (p50_boot - A) sur σ : {s['sigma']['p50'] - sigA:+.4f}")
    print(f"  Écart (p50_boot - A) sur μ : {s['retour']['p50'] - muA:+.4f}")

    # ---------------- Q3 ----------------
    print("\n" + "-" * 86)
    print("Q3 — Portefeuille tangent (non-contraint) et ratio de Sharpe")
    q3A = A["Q3"]
    q3 = r["Q3_tangency_non_contraint"]

    # Partie A
    print("\nPartie A (échantillon original) :")
    print(f"  Moyenne du portefeuille tangent μ_tan (%/an): {q3A['mu']:.6f}")
    print(f"  Variance du portefeuille tangent Var_tan: {q3A['var']:.6f}")
    print(f"  Ecart-type du portefeuille tangent σ_tan (%/an): {q3A['sigma']:.6f}")
    print(f"  Sharpe_tan = (μ_tan - R)/σ_tan: {q3A['sharpe']:.6f}")
    print("\n  Poids par industrie (w_tan):")
    print(q3A["w"].to_string(float_format=lambda x: f"{x:+.6f}"))

    # Bootstrap (médiane + incertitude)
    print("\nBootstrap (médiane p50 + bande 5%-95%) :")
    print(f"  μ_tan (p50) = {q3['mu']['p50']:.6f} | 5%-95%=[{q3['mu']['p05']:.6f}, {q3['mu']['p95']:.6f}]")
    print(f"  Var_tan (p50) = {q3['var']['p50']:.6f} | 5%-95%=[{q3['var']['p05']:.6f}, {q3['var']['p95']:.6f}]")
    print(f"  σ_tan (p50) = {q3['sigma']['p50']:.6f} | 5%-95%=[{q3['sigma']['p05']:.6f}, {q3['sigma']['p95']:.6f}]")
    print(f"  Sharpe (p50) = {q3['sharpe']['p50']:.6f} | 5%-95%=[{q3['sharpe']['p05']:.6f}, {q3['sharpe']['p95']:.6f}]")

    print("\n  Poids bootstrap (p50 + 5%-95%) :")
    for ind, st in q3["poids"].items():
        print(f"    {ind}: p50={st['p50']:+.6f} | 5%={st['p05']:+.6f} | 95%={st['p95']:+.6f}")

    # ---------------- Q4 ----------------
    print("\n" + "-" * 86)
    print("Q4 — Frontière efficiente (sans actif sans risque, no-short)")
    s = r["Q4_point_median"]
    muA = A["Q4_mid"]["mu"]; sigA = A["Q4_mid"]["sigma"]

    print("Point médian de frontière (index ~50%) — Bootstrap vs Partie A :")
    print(f"  σ_boot (p50) = {s['sigma']['p50']:.4f} | 5%-95% = [{s['sigma']['p05']:.4f}, {s['sigma']['p95']:.4f}]"
          f" | σ_A = {sigA:.4f}")
    print(f"  μ_boot (p50) = {s['retour']['p50']:.4f} | 5%-95% = [{s['retour']['p05']:.4f}, {s['retour']['p95']:.4f}]"
          f" | μ_A = {muA:.4f}")

    print("Incertitude (bootstrap) :")
    print(f"  Largeur bande 90% sur σ : {largeur_90(s['sigma']):.4f}")
    print(f"  Largeur bande 90% sur μ : {largeur_90(s['retour']):.4f}")
    print(f"  Écart (p50_boot - A) sur σ : {s['sigma']['p50'] - sigA:+.4f}")
    print(f"  Écart (p50_boot - A) sur μ : {s['retour']['p50'] - muA:+.4f}")

    # ---------------- Q6 ----------------
    print("\n" + "-" * 86)
    print("Q6 — Portefeuille tangent (no-short) et ratio de Sharpe")
    q6A = A["Q6"]
    q6 = r["Q6_tangency_no_short"]

    # Partie A
    print("\nPartie A (échantillon original) :")
    print(f"  Moyenne du portefeuille tangent μ_tan (%/an): {q6A['mu']:.6f}")
    print(f"  Variance du portefeuille tangent Var_tan: {q6A['var']:.6f}")
    print(f"  Ecart-type du portefeuille tangent σ_tan (%/an): {q6A['sigma']:.6f}")
    print(f"  Sharpe_tan = (μ_tan - R)/σ_tan: {q6A['sharpe']:.6f}")
    print("\n  Poids par industrie (w_tan) [no-short]:")
    print(q6A["w"].to_string(float_format=lambda x: f"{x:+.6f}"))

    # Bootstrap
    print("\nBootstrap (médiane p50 + bande 5%-95%) :")
    print(f"  μ_tan (p50) = {q6['mu']['p50']:.6f} | 5%-95%=[{q6['mu']['p05']:.6f}, {q6['mu']['p95']:.6f}]")
    print(f"  Var_tan (p50) = {q6['var']['p50']:.6f} | 5%-95%=[{q6['var']['p05']:.6f}, {q6['var']['p95']:.6f}]")
    print(f"  σ_tan (p50) = {q6['sigma']['p50']:.6f} | 5%-95%=[{q6['sigma']['p05']:.6f}, {q6['sigma']['p95']:.6f}]")
    print(f"  Sharpe (p50) = {q6['sharpe']['p50']:.6f} | 5%-95%=[{q6['sharpe']['p05']:.6f}, {q6['sharpe']['p95']:.6f}]")

    print("\n  Poids bootstrap (p50 + 5%-95%) [no-short] :")
    for ind, st in q6["poids"].items():
        print(f"    {ind}: p50={st['p50']:+.6f} | 5%={st['p05']:+.6f} | 95%={st['p95']:+.6f}")

    print("\n" + sep)


def b1_plot_2_graphiques(res_b1: dict):
    """
    NOUVELLE version (comme demandé) :
      - on SUPPRIME les 3 anciens plots
      - on produit 4 figures de comparaison (Q1, Q3, Q4, Q6)
        * Point(s) Partie A = sombre (dot ou star)
        * Bootstrap = nuage pâle + couleur différente
    """
    series = res_b1["series"]
    A = res_b1["A"]

    # Courbes A (frontières)
    q1_sig_curve = np.array(A["frontieres"]["Q1_sig"], dtype=float)
    q1_mu_curve  = np.array(A["frontieres"]["Q1_mu"], dtype=float)
    q4_sig_curve = np.array(A["frontieres"]["Q4_sig"], dtype=float)
    q4_mu_curve  = np.array(A["frontieres"]["Q4_mu"], dtype=float)

    # =========================
    # Q1 — Comparaison médiane (A) vs points bootstrap
    # =========================
    plt.figure(figsize=(10, 6))
    plt.plot(q1_sig_curve, q1_mu_curve, linewidth=2, label="Frontière efficiente — Partie A (échantillon original)")
    plt.scatter(series["sig_q1_mid"], series["ret_q1_mid"], alpha=0.25, label="Bootstrap — points médians (B réplications)")
    plt.scatter(A["Q1_mid"]["sigma"], A["Q1_mid"]["mu"], s=120, marker="o", label="Partie A — point médian (référence)")
    plt.xlabel("Risque σ")
    plt.ylabel("Retour μ")
    plt.title("Q1 — Comparaison : point médian de frontière (Partie A vs bootstrap)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # =========================
    # Q3 — Tangency : étoile A vs nuage bootstrap
    # =========================
    plt.figure(figsize=(10, 6))
    plt.plot(q1_sig_curve, q1_mu_curve, linewidth=2, label="Frontière efficiente risquée — Partie A")
    
    # --- filtrage des points bootstrap non finis (Q3) ---
    mask_q3 = np.isfinite(series["sig_q3"]) & np.isfinite(series["mu_q3"])

    plt.scatter(
        series["sig_q3"][mask_q3],
        series["mu_q3"][mask_q3],
        alpha=0.25,
        label="Bootstrap — portefeuilles tangents (Q3)"
    )

    # --- zoom sur la masse centrale (95%) ---
    x = series["sig_q3"][mask_q3]
    y = series["mu_q3"][mask_q3]

    plt.xlim(0, np.quantile(x, 0.95) * 1.05)
    plt.ylim(np.quantile(y, 0.05) * 1.05, np.quantile(y, 0.95) * 1.05)

    plt.scatter(A["Q3"]["sigma"], A["Q3"]["mu"], s=220, marker="*", label="Partie A — portefeuille tangent (Q3)")
    plt.xlabel("Risque σ")
    plt.ylabel("Retour μ")
    plt.title("Q3 — Comparaison : portefeuille tangent (Partie A vs bootstrap)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # =========================
    # Q4 — Comparaison médiane (A) vs points bootstrap
    # =========================
    plt.figure(figsize=(10, 6))
    plt.plot(q4_sig_curve, q4_mu_curve, linewidth=2, linestyle="--", label="Frontière efficiente no-short — Partie A")
    plt.scatter(series["sig_q4_mid"], series["ret_q4_mid"], alpha=0.25, label="Bootstrap — points médians (Q4)")
    plt.scatter(A["Q4_mid"]["sigma"], A["Q4_mid"]["mu"], s=120, marker="o", label="Partie A — point médian (référence)")
    plt.xlabel("Risque σ")
    plt.ylabel("Retour μ")
    plt.title("Q4 — Comparaison : point médian de frontière no-short (Partie A vs bootstrap)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # =========================
    # Q6 — Tangency no-short : étoile A vs nuage bootstrap
    # =========================
    plt.figure(figsize=(10, 6))
    plt.plot(q4_sig_curve, q4_mu_curve, linewidth=2, linestyle="--", label="Frontière efficiente no-short — Partie A")
    plt.scatter(series["sig_q6"], series["mu_q6"], alpha=0.25, label="Bootstrap — portefeuilles tangents (Q6)")
    plt.scatter(A["Q6"]["sigma"], A["Q6"]["mu"], s=220, marker="*", label="Partie A — portefeuille tangent (Q6)")
    plt.xlabel("Risque σ")
    plt.ylabel("Retour μ")
    plt.title("Q6 — Comparaison : portefeuille tangent no-short (Partie A vs bootstrap)")
    plt.legend()
    plt.grid(True)
    plt.show()

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