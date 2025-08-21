import numpy as np
from scipy.optimize import minimize

def elo_map_rating(results, problem_elos, mu0=1500, sigma0=400, eps=0.1):
    """
    Estimate MAP Elo rating for a model under the Bayesian formulation
    described in LiveCodeBench Pro.

    results: list of 0/1 (accepted=1, rejected=0)
    problem_elos: list of problem difficulty ratings
    mu0: prior mean for rating
    sigma0: prior standard deviation
    eps: step size for curvature approximation
    """
    y = np.array(results)
    d = np.array(problem_elos)

    # Probability of solving problem i given rating r
    def pi(r):
        return 1.0 / (1.0 + 10.0 ** ((d - r) / 400.0))

    # Log-posterior (unnormalized)
    def L(r):
        P = pi(r)
        ll = np.sum(y * np.log(P + 1e-12) + (1 - y) * np.log(1 - P + 1e-12))
        prior = -((r - mu0) ** 2) / (2 * sigma0 ** 2)
        return ll + prior

    # Negative log-posterior for optimization
    def negL(r):
        return -L(r[0])

    # Optimize
    res = minimize(negL, x0=np.array([mu0]), method="L-BFGS-B")
    r_hat = res.x[0]

    # Approximate curvature at r_hat
    L_plus = L(r_hat + eps)
    L_mid = L(r_hat)
    L_minus = L(r_hat - eps)

    I = -(L_plus - 2 * L_mid + L_minus) / (eps ** 2)

    std = 1.0 / np.sqrt(I) if I > 0 else np.inf

    return r_hat, std

# Example:
results = [1, 0, 0, 0, 0, 0, 0, 0, 1]
problems = [900, 2000, 1100, 1300, 2900, 3100, 1900, 1500, 3000]

r_hat, std = elo_map_rating(results, problems)
print(f"Estimated rating: {r_hat:.2f} ± {std:.2f}")