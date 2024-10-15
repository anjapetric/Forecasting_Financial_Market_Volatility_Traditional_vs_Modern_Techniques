# main.py
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.optimize import minimize
from filter_egarch_asymmetric import FilterEGARCHAsymmetric
from negative_log_likelihood_egarch_asymmetric import NegativeLogLikelihoodEGARCHAsymmetric

# Load data
data = scipy.io.loadmat('data.mat')
returns = data['returns'].flatten()
dates = data['dates'].flatten()

# Convert dates to Python datetime format
dates = [datetime.strptime(str(int(date)), '%Y%m%d') for date in dates]

# Set starting values
starting_values = [np.mean(returns), np.log(np.var(returns)/20), 0.8, 0.1, 0.05, 6]

# Calculate the negative log likelihood with starting values
nll_calculator = NegativeLogLikelihoodEGARCHAsymmetric()
initial_nll = nll_calculator(starting_values, returns)
print(f"Initial Negative Log Likelihood: {initial_nll}")

# Set optimization options
options = {
    'maxiter': 1000000,
    'disp': True
}

# Set bounds
bounds = [(-np.inf, np.inf), (-np.inf, np.inf), (-1, 1), (-np.inf, np.inf), (-np.inf, np.inf), (2, 40)]

# Perform optimization
result = minimize(nll_calculator, starting_values, args=(returns,), bounds=bounds, options=options, method='L-BFGS-B')
ml_parameters = result.x
negative_log_likelihood1 = result.fun

mu_hat, lambda_hat, phi_hat, kappa_hat, kappa_tilde_hat, nu_hat = ml_parameters
filter_instance = FilterEGARCHAsymmetric()
sigma_squared, lamdba, epsilon, u, v = filter_instance(mu_hat, lambda_hat, phi_hat, kappa_hat, kappa_tilde_hat, nu_hat, returns)

# Calculate NIC
epsilon_egarch = epsilon
nic_egarch = 2 * kappa_hat * u + 2 * kappa_tilde_hat * v
plt.scatter(epsilon_egarch, nic_egarch, c='k', marker='o')
plt.xlabel('News shock $\varepsilon_t$', fontsize=15)
plt.ylabel('Impact on volatility', fontsize=15)
plt.grid(True)
plt.show()

# Calculate AIC and BIC
num_parameters = 6
n_obs = len(returns)
aic = 2 * num_parameters + 2 * negative_log_likelihood1
bic = np.log(n_obs) * num_parameters + 2 * negative_log_likelihood1

# Check implied shocks
mean_epsilon = np.mean(epsilon_egarch)
var_epsilon = np.var(epsilon_egarch)
print(f"Mean of epsilon: {mean_epsilon}, Variance of epsilon: {var_epsilon}")

# Display estimated parameters
print('Estimated parameters of the asymmetric Beta-t-EGARCH model:')
print(f'mu_hat           = {mu_hat:.4f}')
print(f'lambda_hat       = {lambda_hat:.4f}')
print(f'phi_hat          = {phi_hat:.4f}')
print(f'kappa_hat        = {kappa_hat:.4f}')
print(f'kappa_tilde_hat  = {kappa_tilde_hat:.4f}')
print(f'nu_hat           = {nu_hat:.4f}')
print(f'Log Likelihood           = {-negative_log_likelihood1:.4f}')
print(f'AIC           = {aic:.4f}')
print(f'BIC           = {bic:.4f}')
