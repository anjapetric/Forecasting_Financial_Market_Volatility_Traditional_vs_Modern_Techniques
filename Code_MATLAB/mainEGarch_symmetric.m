%% Clear all data 
clear
close all

%% Load the directory (change this to where you have saved this file)
%pad = 'C:/Users/61849rla/Dropbox/Basis Weco/';
pad = 'C:\Users\613746br\OneDrive - Erasmus University Rotterdam\Documents\Uni\bsc2\year 3\Block 5\Introductary seminar cases\QF\matlab';
cd(pad);
load('data')

%% Get dates in correct Matlab format
dates_str = cellstr(num2str(dates));
dates     = datetime(dates_str, 'InputFormat', 'yyyyMMdd');

%% Estimate a beta-t-eGARCH model
format short
clear  NegativeLogLikelihood_EGARCH

%% Set starting values for the optimisation and check value of the objective

% Starting values
% [mu, lambda, phi, kappa, kappa_tilde, nu]
startingvalues = [mean(returns); log(var(returns)/20); 0.8; 0.1; 6];

% Get the negative log likelihood at the starting values (the optimiser
% should beat this!) 
NegativeLogLikelihood_EGARCH_symmetric(startingvalues,returns)

%% Clear any pre-existing options
clearvars options

% Load some options (no need to change this)
options  =  optimset('fmincon');
options  =  optimset(options , 'TolFun'      , 1e-6);
options  =  optimset(options , 'TolX'        , 1e-6);
options  =  optimset(options , 'Display'     , 'on');
options  =  optimset(options , 'Diagnostics' , 'on');
options  =  optimset(options , 'LargeScale'  , 'off');
options  =  optimset(options , 'MaxFunEvals' , 10^6) ;
options  =  optimset(options , 'MaxIter'     , 10^6) ;


%% Parameter lower bound and upper bound
% [mu, lambda, phi, kappa, kappa_tilde, nu]
lowerbound = [-inf, -inf, -1, -inf, 2]; % phi can range between -1 and 1, nu > 2
upperbound = [inf, inf, 1, inf, 40]; % upper bound for nu is 40

%% Do the actual optimisation (this should be very fast, less than a second)
tic
[ML_parameters,NegativeLogLikelihood1]=fmincon('NegativeLogLikelihood_EGARCH_symmetric', startingvalues ,[],[],[],[],lowerbound,upperbound,[],options,returns);
toc

%% Save the parameters and compute GARCH filter at these parameters

mu_hat = ML_parameters(1);
lambda_hat = ML_parameters(2);
phi_hat = ML_parameters(3);
kappa_hat = ML_parameters(4);
nu_hat = ML_parameters(5);
[sigmasquared, lamdba, epsilon, u,] = Filter_EGARCH_symmetric(mu_hat, lambda_hat, phi_hat, kappa_hat, nu_hat, returns);

%% calculating the AIC and BIC
[aic,bic] = aicbic(-NegativeLogLikelihood1, 5, length(returns));
%% calculating NIC
epsilon_EGARCH = epsilon;
NIC_EGARCH = 2*kappa_hat.*u;
figure
scatter(epsilon_EGARCH,NIC_EGARCH,'ko')
xlabel('News shock $\varepsilon_t$','Interpreter','latex')
ylabel('Impact on volatility','Interpreter','latex')
set(gca, 'FontName', 'Times', 'fontsize', 20, 'TickDir', 'out');


%% Check that the implied shocks have approximately the desired characteristics (mean zero, variance one)
disp([mean(epsilon_EGARCH);var(epsilon_EGARCH)])
% should be roughly [0;1]
%% Display the estimated parameter values
fprintf('Estimated parameters of the symmetric Beta-t-EGARCH model:\n');
fprintf('mu_hat           = %.4f\n', mu_hat);
fprintf('lambda_hat       = %.4f\n', lambda_hat);
fprintf('phi_hat          = %.4f\n', phi_hat);
fprintf('kappa_hat        = %.4f\n', kappa_hat);
fprintf('nu_hat           = %.4f\n', nu_hat);
fprintf('Log Likelihood    = %.4f\n', -NegativeLogLikelihood1)      
fprintf('aic           = %.4f\n', aic);
fprintf('bic           = %.4f\n', bic);
