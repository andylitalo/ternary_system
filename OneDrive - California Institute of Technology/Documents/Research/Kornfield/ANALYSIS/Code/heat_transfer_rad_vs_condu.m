% Calculates the radiative and conductive heat transfers and compares their
% magnitude for polyurethane foam with cyclopentane as a blowing agent at
% 300 K
clear; close all; clc;

% Parameters
sigma = 5.67e-8; % Stefan-Boltzman constant, W/m^2.K^4
K = 1500; % extinction coefficient, 1/m [Glicksman 1994]
delta = 0.97; % void fraction [Dow]
kg = 0.01; % thermal conductivity of gas (cyclopentane), W/m.K [Vassiliou 2015]
kp = 0.237; % thermal conductivity of bulk solid (polyurethane), W/m.K [Iqbal 2012]
fs = 0.85; % fraction of polymer in struts [Reitz 1984]
T = 300; % temperature, K

% Radiation thermal conductivity [Glicksman 1994]
krad = 16/(3*K)*sigma*T^3; % [W/m.K]
% Conduction thermal conductivity (Rosseland Eqn) [Glicksman 1994]
kcond = delta*kg + kp*(1-delta)/3*(2-fs); % [W/m.K]

% radiation should account for 20-30% of heat conduction in polymeric foams
% (Glicksman 1994). Closer to 20% for PU (Tom Fitzgibbons, Dow).
% "In a typical low-density closed cell polymeric foam filled with a low
% conductivity gas, heat transfer through the gas comprises 40% to 50% of
% the total heat transfer, while radiation and solid conduction make up the
% balance in roughly equal proportions" p. 104 of Glicksman 1994.
fprintf(strcat('Radiation to conduction thermal conductivity kr/kc',...
    '= %.3f.\nShould be ~0.4-0.5.\n'),krad/kcond)
fprintf(strcat('Radiation makes up %.0f%% of heat transfer.\n',...
    'Should be about 25%%.\n'),100*krad/(2*kcond)) % based on above quote