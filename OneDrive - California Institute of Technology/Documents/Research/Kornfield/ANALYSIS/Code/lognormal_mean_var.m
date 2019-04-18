% LOGNORMAL_MEAN_VAR computes the mu and sigma parameters to produce a
% log-normal distribution with the given mean and variance.
clear; close all; clc;

% Mean and variance
M = 0.50; V = [0.5 5 50 500 5000 1e4]/100;

% mu and sigma
mu = log(M.^2./sqrt(V + M.^2));
sigma = sqrt(log(V./M.^2+1));

% Plot
for i = 1:length(V)
    X = linspace(1,100)/100;
    Y = lognpdf(X, mu(i), sigma(i));
    figure()
    loglog(X,Y)
    axis([0.01 1 1e-4 10])
end