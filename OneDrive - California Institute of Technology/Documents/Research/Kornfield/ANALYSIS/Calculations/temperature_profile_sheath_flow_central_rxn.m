% TEMPERATURE_PROFILE_... computes the increase in temperature for fully
% developed flow after 100 ms as well as the estimated temperature profile
% starting from a constant temperature.


% System parameters
DHrxn = -24e3*4.184;
nNCO = 2;
rhoIso = 1230;
rhoPoly = 1018;
Mw = 0.286;
L = 0.1;
tau = 0.1;
RIso = 1e-5;
RPoly = 25e-5;
CpIso = 430*4.184;
CpPoly = 497*4.184;
kPoly = 0.126;
kIso = 0.0003*4.184*100;
V = 0.5;
Vi = 2*V;
Vo = V;
T0 = 301;

% Plot parameters
LW = 3;
A_FS = 16;
T_FS = 20;
AX_FS = 14;
xmin = 0;
xmax = RPoly*1e6; % radius of channel [um]
ymin = 300.85; % [K]
ymax = 303.5; % [K]

% Derived parameters
VIso = pi*RIso^2*L;
mIso = rhoIso*VIso;
nMol = mIso/Mw;
q = -nMol*nNCO*DHrxn;
W = q/(VIso*tau);
Phi = 1/2 * (W*RIso*(RPoly^2-RIso^2)*Vo*rhoPoly*CpPoly)/...
    (rhoIso*CpIso*Vi*RIso^2+rhoPoly*CpPoly*Vo*(RPoly^2-RIso^2));
Ti = @(r,z,T0) T0 + W*(1-2/(W*RIso)*Phi)*V/Vi*z./(rhoIso*CpIso*V)+W*RPoly^2/...
    (4*kIso)*((2*(1-2/(W*RIso)*Phi)*V/Vi-1)*((r/RPoly).^2-(RIso/RPoly)^2)-...
    (1-2/(W*RIso)*Phi)*V/Vi/2*((r/RPoly).^4-(RIso/RPoly)^4));
To = @(r,z,T0) T0 + (2*RIso*Phi)./((RPoly^2-RIso^2)*Vo*rhoPoly*CpPoly)*z+...
    (RIso*RPoly^2*Phi*V)./((RPoly^2-RIso^2)*V*kPoly)*(-(r.^4-RIso^4)./(4*RPoly^4)+...
    (r.^2-RIso^2)./RPoly^2 - log(r/RIso));

% Computations
rIso = linspace(0,RIso);
rPoly = linspace(RIso, RPoly);
r = 1e6*[rIso, rPoly];
TIso0 = Ti(rIso,0,T0);
TPoly0 = To(rPoly,0,T0);
T0List = [TIso0, TPoly0];
TIsoL = Ti(rIso,L,T0);
TPolyL = To(rPoly,L,T0);
TLList = [TIsoL, TPolyL];
% estimate fraction of flow developed
alphaPoly = kPoly/(rhoPoly*CpPoly); % thermal diffusivity of polyol
distanceWalls = RPoly-RIso; % length for heat to conduct to reach walls
tauThermal = distanceWalls^2/alphaPoly;
fracDev = tau / tauThermal; % fraction developed is fraction of thermal time
% Estimated temperature profile
TEstList = T0 + (TLList-T0List) + fracDev*(T0List-min(T0List));

% Plot of estimated development of flow
figure()
plot(r, T0*ones(1,length(r)),'b--','LineWidth',LW)
hold on
grid on
plot(r,TEstList,'r','LineWidth',LW)
% plot interface
plot(1e6*[RIso,RIso],[ymin ymax],'g--','LineWidth',LW)
xlabel('r [\mum]','FontSize',A_FS)
ylabel('T [K]','FontSize',A_FS)
set(gca,'FontSize', AX_FS)
legend('z = 0 (0 ms)','z = 10 cm (100 ms)', 'R_{interf} = 10 \mum')
title('Estimated Temperature during Polyurethane Reaction', 'FontSize', T_FS)
axis([xmin xmax ymin ymax])

% Plot of increase in temperature of fully developed flow
figure()
plot(r,T0List,'LineWidth',LW)
hold on
grid on
plot(r,TLList,'r','LineWidth',LW)
% plot interface
plot(1e6*[RIso,RIso],ylim,'g--','LineWidth',LW)
xlabel('r [\mum]','FontSize',A_FS)
ylabel('T [K]','FontSize',A_FS)
set(gca,'FontSize', AX_FS)
legend('z = 0 (0 ms)','z = 10 cm (100 ms)', 'R_{interf} = 10 \mum')
title('Fully Developed Temperature during Polyurethane Reaction', 'FontSize', T_FS)

% Plot of just fully developed flow
figure()
plot(r,TLList + 301 - min(TLList),'r','LineWidth',LW)
hold on
grid on
% plot interface
plot(1e6*[RIso,RIso],ylim,'g--','LineWidth',LW)
xlabel('r [\mum]','FontSize',A_FS)
ylabel('T [K]','FontSize',A_FS)
set(gca,'FontSize', AX_FS)
legend('Fully developed temp.', 'R_{interf} = 10 \mum')
title('Fully Developed Temperature during Polyurethane Reaction', 'FontSize', T_FS)
