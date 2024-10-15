function [lambda, sigma, epsilon, u, v] = Filter_EGARCH_asymmetric(mu, initial_lambda, phi, kappa, kappa_tilde, nu, returns)
    % Extract the sample size
    T = size(returns, 1);

    % Prefill the variables we are going to track
    u = zeros(T,1);
    v = zeros(T,1);
    epsilon = zeros(T,1);
    lambda = zeros(T, 1);
    sigma = zeros(T, 1);
    
    % Initialize at the unconditional mean of lambda
    lambda_start = initial_lambda;
    sigma_start = exp(lambda_start);

    % Run the Beta-t-EGARCH filter
    for t = 1:T
        if t==1
            sigma(t) = sigma_start;
            lambda(t) = initial_lambda;
            epsilon(t) = (returns(t) - mu)*exp(-lambda(t));
            u(t) = sqrt((nu + 3) / (2 * nu)) * (((nu + 1) / (nu - 2 + epsilon(t)^2)) *epsilon(t)^2 - 1);
            v(t) = sqrt((nu - 2) * (nu + 3) / (nu * (nu + 1))) * ((nu + 1) / (nu - 2 + epsilon(t)^2)) * epsilon(t);
        % Compute residual
        else 
            lambda(t) = initial_lambda * (1 - phi) + phi * lambda(t-1) + kappa * u(t-1) + kappa_tilde * v(t-1);
            sigma(t) = exp(lambda(t));
            epsilon(t) = (returns(t) - mu)*exp(-lambda(t));
            u(t) = sqrt((nu + 3) / (2 * nu)) * (((nu + 1) / (nu - 2 + epsilon(t)^2)) *epsilon(t)^2 - 1);
            v(t) = sqrt((nu - 2) * (nu + 3) / (nu * (nu + 1))) * ((nu + 1) / (nu - 2 + epsilon(t)^2)) * epsilon(t);
        end
        
    end
end
