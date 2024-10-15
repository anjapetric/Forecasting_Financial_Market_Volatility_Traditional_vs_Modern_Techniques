function [output]=studentpdf(epsilon,nu)
output = gamma((nu + 1)/2)/gamma(nu/2)/sqrt(pi*(nu-2))*(1+epsilon.^2/(nu-2)).^(-(nu+1)/2);
end







