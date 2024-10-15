function [ sigmasquared ] = Filter_GARCH(mu,omega,alpha,beta,returns)
% Extract the sample size (make sure returns are a column vector)
T     = size(returns,1);
% Prefill the variable that we are going to track
sigmasquared = zeros(T,1);
% Define sigmabarsquared 
averagesigmasquared = omega/(1-alpha-beta);
% Run the GARCH filter
for t=1:T
    if t==1
        % Initialise at the unconditional mean of sigmasquared
        sigmasquared(t,1)     = averagesigmasquared;      
   else
        sigmasquared(t,1)    = omega + alpha * (returns(t-1,1)-mu)^2 + beta * sigmasquared(t-1,1);
    end
end
% Close the function
end

