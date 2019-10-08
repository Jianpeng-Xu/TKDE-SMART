function LossOverTimeSMARTb(responseIndex, numStartStations, stationInitScheme, randomSeeds, R, lambda, beta)
% This function is to record the loss over time for this SMART-b using the
% best parameters achieved from the tuning.
rng(randomSeeds);
addpath('../utils/');

% responseIndex:
% 1 : tmax; 2: tmin; 3: tmean; 4: precip
% stationInitScheme: 1 - random initilize index, 2 - use cluster centroids
% after do a clustering method. This index is returned from a function
% called getStationInitIndex

% Useful variables:
% deszXMonth: 1118 * 371 * 13
% deszYMonth: 1118 * 371 * 4
dataset = 'deseasonedMonthly_smallZscore_new.mat';
path = '../../dataGSOD/';
load([path dataset]);

[S, T, d] = size(deszXMonth);
deszYMonth = deszYMonth(1:S, 1:T, :);

[InitialStations, addStations] = getStationInitIndex(stationLat, stationLon, numStartStations, stationInitScheme, randomSeeds);

deszXMonth = deszXMonth([InitialStations; addStations], :, :);
deszYMonth = squeeze(deszYMonth([InitialStations; addStations], :, responseIndex));

TrainingSize = 120;
ValidationSize = 120;
TestingSize = T - TrainingSize - ValidationSize;

MAE_ALL_time = NaN(1,T);

Y_hat = NaN(S, T);
pAll = ones(S+T - numStartStations - 1, 1);
pAll(randperm(length(pAll), T-1)) = 0;
pAll = [pAll;0]; % manually set the last update is time

% run learning method
% do incremental learning.
% randomly choose whether to incremental over time or space
Y_hat_local = NaN(S, T);
spaceIndex = (1 : length(InitialStations))';
timeIndex = [];
% Initialize regularization coefficients
lambda_local =lambda;
beta_local = beta;
R_local = R;

t = 0;
s = length(spaceIndex);

pIndex = 1;
while s <S || t < T
    p = pAll(pIndex);
    pIndex = pIndex + 1;
    tos = ' ';
    if p <=0.3
        tos = 'time';
    else
        tos = 'space';
    end
    
    if s >= S || t == 0
        tos = 'time';
    end
    if t >= T
        tos = 'space';
    end
    if strcmp(tos, 'time')
        % incremental over time
        % increase t
        t = t + 1;
        timeIndex = 1:t;
        % prepare data
        X_T = deszXMonth(spaceIndex, timeIndex, :);
        Y_T = deszYMonth(spaceIndex, timeIndex(1:t-1)); % remove the last element in training
        % update models
        [W, V, A, B, C] = wisdom_sparsa(X_T, Y_T, lambda_local, beta_local, R_local);
        % do prediction on X_t only
        X_t = squeeze(X_T(:, t, :));
        Y_t = squeeze(deszYMonth(spaceIndex, timeIndex(t)));
        B_t = squeeze(B(t,:))';
        Y_hat_local(spaceIndex,t) = (sum(X_t .* bsxfun(@plus, A * W, B_t'*V),2));
        MAE_local = mean(abs(Y_hat_local(spaceIndex,t) - Y_t));
        % record the loss
        MAE_ALL_time(t) = MAE_local;

        fprintf([num2str(t), ' ']);
        %             fprintf(['MAE = ' num2str(MAE_local) ', lambda = ' num2str(lambda_local)...
        %                 ', beta = ' num2str(beta_local) '\n']);
    elseif strcmp(tos, 'space')
        % incremental over space
        %             fprintf('Incremental over space\n');
        % incease s
        s = s + 1;
        % update space index
        newStationIndex = s;
        spaceIndex = [spaceIndex; newStationIndex];
        % do nothing: no test is performed with only new station
    end
    
end

% save the result 
save(['LossOverTimeSMARTb-'  num2str(responseIndex) '-' num2str(numStartStations) '-' num2str(stationInitScheme)  '-' num2str(R) '-' num2str(lambda) '-' num2str(beta) '-' num2str(randomSeeds) '.mat'], ...
    'MAE_ALL_time');