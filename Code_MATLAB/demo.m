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

%% Plot log return
figure
plot(dates, returns, 'k')
ylabel('$r_t$', 'Interpreter', 'latex')
hold on

% Define tick locations and format
tick_dates = datetime({'2000-01-01'; '2001-01-01'; '2002-01-01'; '2003-01-01'; '2004-01-01'; '2005-01-01'; '2006-01-01'; '2007-01-01'; '2008-01-01'; '2009-01-01'; '2010-01-01'; '2011-01-01'; '2012-01-01'; '2013-01-01'; '2014-01-01'; '2015-01-01'; '2016-01-01'; '2017-01-01'; '2018-01-01'; '2019-01-01'; '2020-01-01'; '2021-01-01'; '2022-01-01'; '2023-01-01'; '2024-01-01'}, 'InputFormat', 'yyyy-MM-dd');
xticks(tick_dates);
dateFormat = 'yy';
xtickformat(dateFormat);
xtickangle(0);

% Set axis limits and properties
xlim([datetime('2000-01-01'), datetime('2025-01-01')]);
ylim([-15, 15]);
set(gca, 'FontName', 'Times', 'fontsize', 20, 'TickDir', 'out');

%% Plot the VIX (Figure 1 in case study)
figure
plot(dates,vix,'k')

% Define tick locations and format
tick_dates = datetime({'2000-01-01'; '2001-01-01'; '2002-01-01'; '2003-01-01'; '2004-01-01'; '2005-01-01'; '2006-01-01'; '2007-01-01'; '2008-01-01'; '2009-01-01'; '2010-01-01'; '2011-01-01'; '2012-01-01'; '2013-01-01'; '2014-01-01'; '2015-01-01'; '2016-01-01'; '2017-01-01'; '2018-01-01'; '2019-01-01'; '2020-01-01'; '2021-01-01'; '2022-01-01'; '2023-01-01'; '2024-01-01'}, 'InputFormat', 'yyyy-MM-dd');
xticks(tick_dates);
dateFormat = 'yy';
xtickformat(dateFormat);
xtickangle(0);

% Set axis limits and properties
xlim([datetime('2000-01-01'), datetime('2025-01-01')]);
ylim([0, 100]);
set(gca, 'FontName', 'Times', 'fontsize', 20, 'TickDir', 'out');
title('VIX')

%% Reproduce Figure 2 in case study
figure
scatter(diff(vix),returns(2:end),'ok')
set(gca,'FontName','Times','fontsize',20)
set(gca,'TickDir','out')
xlabel('$\Delta \mbox{VIX}_t$','Interpreter','latex')
ylabel('$r_t$','Interpreter','latex')

%% Estimate a simple t-GARCH model

format short
clear  NegativeLogLikelihood_GARCH

%% Set starting values for the optimisation and check value of the objective

% Starting values
% startingvalues = [mu,sigma2,alpha,beta,nu]
startingvalues=[mean(returns);var(returns)/20;0.10;0.88;6];

% Get the negative log likelihood at the starting values (the optimiser
% should beat this!) 
NegativeLogLikelihood_GARCH(startingvalues,returns)

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
% parameter = [mu,omega,alpha,beta,nu]
lowerbound = [-inf,0,0,0,2]; % mu can be anything, omega, alpha, beta should be positve, we want nu>2
upperbound = [inf,inf,inf,inf,40]; % upper bound for nu is 40, which is high enough to resemble a normal distribution

%% Do the actual optimisation (this should be very fast, less than a second)
tic
[ML_parameters,NegativeLogLikelihood1]=fmincon('NegativeLogLikelihood_GARCH', startingvalues ,[],[],[],[],lowerbound,upperbound,[],options,returns);
toc

%% Save the parameters and compute GARCH filter at these parameters

mu_hat           = ML_parameters(1);
omega_hat        = ML_parameters(2);
alpha_hat        = ML_parameters(3);
beta_hat         = ML_parameters(4);
nu_hat           = ML_parameters(5);
[ sigmasquared ] = Filter_GARCH(mu_hat,omega_hat,alpha_hat,beta_hat,returns);

%% Plot the data along with the output of the GARCH filter
figure
plot(dates,returns,'k')
hold on
plot(dates,sqrt(sigmasquared),'r:','linewidth',2)

% Define tick locations and format
tick_dates = datetime({'2000-01-01'; '2001-01-01'; '2002-01-01'; '2003-01-01'; '2004-01-01'; '2005-01-01'; '2006-01-01'; '2007-01-01'; '2008-01-01'; '2009-01-01'; '2010-01-01'; '2011-01-01'; '2012-01-01'; '2013-01-01'; '2014-01-01'; '2015-01-01'; '2016-01-01'; '2017-01-01'; '2018-01-01'; '2019-01-01'; '2020-01-01'; '2021-01-01'; '2022-01-01'; '2023-01-01'; '2024-01-01'}, 'InputFormat', 'yyyy-MM-dd');
xticks(tick_dates);
dateFormat = 'yy';
xtickformat(dateFormat);
xtickangle(0);

% Set axis limits and properties
xlim([datetime('2000-01-01'), datetime('2025-01-01')]);
ylim([-15,15]);
set(gca, 'FontName', 'Times', 'fontsize', 20, 'TickDir', 'out');
legend('data','Estimated $\sigma_{t|t-1}$ from GARCH model','Interpreter','latex')

%% This graph shows 3 measures of volatility: GARCH output, sqrt(rv5), and rescaled VIX
figure
plot(dates,returns.^2,'ok')
hold on 
plot(dates,rv5,'b','linewidth',3)
hold on 
plot(dates,(1/250)*vix.^2,'y','linewidth',3)
hold on
plot(dates,sigmasquared,'r:','linewidth',2)

% Define tick locations and format
tick_dates = datetime({'2000-01-01'; '2001-01-01'; '2002-01-01'; '2003-01-01'; '2004-01-01'; '2005-01-01'; '2006-01-01'; '2007-01-01'; '2008-01-01'; '2009-01-01'; '2010-01-01'; '2011-01-01'; '2012-01-01'; '2013-01-01'; '2014-01-01'; '2015-01-01'; '2016-01-01'; '2017-01-01'; '2018-01-01'; '2019-01-01'; '2020-01-01'; '2021-01-01'; '2022-01-01'; '2023-01-01'; '2024-01-01'}, 'InputFormat', 'yyyy-MM-dd');
xticks(tick_dates);
dateFormat = 'yy';
xtickformat(dateFormat);
xtickangle(0);

% Set axis limits and properties
xlim([datetime('2000-01-01'), datetime('2025-01-01')]);
ylim([0,100]);
set(gca, 'FontName', 'Times', 'fontsize', 20, 'TickDir', 'out');

% Give legend and title
legend('Squared returns', 'Realised variance','Rescaled version of VIX$^2$','Estimated $\sigma^2_{t|t-1}$ from GARCH model','Interpreter','latex')
title('Variance of S&P500 returns by different measures')

%% Compute news impact function for GARCH
epsilon_GARCH   = ( returns - mu_hat ) ./ sqrt(sigmasquared);
NIC_GARCH       = alpha_hat * (epsilon_GARCH.^2-1);
figure
scatter(epsilon_GARCH,NIC_GARCH,'ko')
xlabel('News shock $\varepsilon_t$','Interpreter','latex')
ylabel('Impact on volatility','Interpreter','latex')
set(gca, 'FontName', 'Times', 'fontsize', 20, 'TickDir', 'out');

%% Check that the implied shocks have approximately the desired characteristics (mean zero, variance one)
disp([mean(epsilon_GARCH);var(epsilon_GARCH)])
% should be roughly [0;1]


