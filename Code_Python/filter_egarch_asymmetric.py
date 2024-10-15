# filter_egarch_asymmetric.py
import numpy as np
from scipy.special import gamma

class FilterEGARCHAsymmetric:
    def __call__(self, mu, initial_lambda, phi, kappa, kappa_tilde, nu, returns):
        T = len(returns)

        u = np.zeros(T)
        v = np.zeros(T)
        epsilon = np.zeros(T)
        lamdba = np.zeros(T)
        sigma = np.zeros(T)

        lamdba[0] = initial_lambda
        sigma[0] = np.exp(lamdba[0])

        for t in range(1, T):
            epsilon[t-1] = (returns[t-1] - mu) * np.exp(-lamdba[t-1])
            u[t-1] = np.sqrt((nu + 3) / (2 * nu)) * (((nu + 1) / (nu - 2 + epsilon[t-1] ** 2)) * epsilon[t-1] ** 2 - 1)
            v[t-1] = np.sqrt((nu - 2) * (nu + 3) / (nu * (nu + 1))) * ((nu + 1) / (nu - 2 + epsilon[t-1] ** 2)) * epsilon[t-1]

            lamdba[t] = initial_lambda * (1 - phi) + phi * lamdba[t-1] + kappa * u[t-1] + kappa_tilde * v[t-1]
            sigma[t] = np.exp(lamdba[t])

        epsilon[-1] = (returns[-1] - mu) * np.exp(-lamdba[-1])
        u[-1] = np.sqrt((nu + 3) / (2 * nu)) * (((nu + 1) / (nu - 2 + epsilon[-1] ** 2)) * epsilon[-1] ** 2 - 1)
        v[-1] = np.sqrt((nu - 2) * (nu + 3) / (nu * (nu + 1))) * ((nu + 1) / (nu - 2 + epsilon[-1] ** 2)) * epsilon[-1]

        return sigma, lamdba, epsilon, u, v
