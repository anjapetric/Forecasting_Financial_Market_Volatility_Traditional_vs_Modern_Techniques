# negative_log_likelihood_egarch_asymmetric.py
import numpy as np
from scipy.special import gamma
from filter_egarch_asymmetric import FilterEGARCHAsymmetric

class NegativeLogLikelihoodEGARCHAsymmetric:
    def student_t_pdf(self, x, nu):
        const = gamma((nu + 1) / 2) / (np.sqrt(np.pi * (nu - 2)) * gamma(nu / 2))
        return const * (1 + x**2 / (nu - 2))**(-(nu + 1) / 2)

    def __call__(self, parameter_vector, returns):
        mu, initial_lambda, phi, kappa, kappa_tilde, nu = parameter_vector

        filter_instance = FilterEGARCHAsymmetric()
        _, sigma, epsilon, _, _ = filter_instance(mu, initial_lambda, phi, kappa, kappa_tilde, nu, returns)

        ll = -np.log(sigma) + np.log(self.student_t_pdf(epsilon, nu))
        negative_ll = -np.sum(ll)

        return negative_ll
