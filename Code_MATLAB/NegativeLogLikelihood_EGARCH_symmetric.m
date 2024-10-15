function [negativeLL] = NegativeLogLikelihood_EGARCH_symmetric(parameter_vector, returns)
    % Extract the parameters
    mu = parameter_vector(1,1);
    lambda = parameter_vector(2,1);
    phi = parameter_vector(3,1);
    kappa = parameter_vector(4,1);
    nu = parameter_vector(5,1);

    % Run the Beta-t-EGARCH filter to get lambda and sigma
    [~, sigma, epsilon, ~] = Filter_EGARCH_symmetric(mu, lambda, phi, kappa, nu, returns);

    % Compute the implied residuals

    % Compute the log-likelihood of each observation
    LL = -log(sigma) + log(studentpdf(epsilon, nu));
    % Negative log likelihood sum (Equation 4.2 in case study)
    negativeLL = -sum(LL);

    % Close the function
end
