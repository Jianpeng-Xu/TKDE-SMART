function WISDOMT(responseIndex, numStartStations, stationInitScheme, randomSeeds,  R, lambda, eta, beta)
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
path = '../../data/';
load([path dataset]);

[S, T, d] = size(deszXMonth);
deszYMonth = deszYMonth(1:S, 1:T, :);

% Prepare the data
if responseIndex == 0
    responseIndex = 4; % 4 for precipitation - default
end

[InitialStations, addStations] = getStationInitIndex(stationLat, stationLon, numStartStations, stationInitScheme, randomSeeds);

deszXMonth = deszXMonth([InitialStations; addStations], :, :);
deszYMonth = squeeze(deszYMonth([InitialStations; addStations], :, responseIndex));

TrainingSize = 120;
ValidationSize = 120;
TestingSize = T - TrainingSize - ValidationSize;

pAll = ones(S+T - numStartStations - 1, 1);
pAll(randperm(length(pAll), T-1)) = 0;
pAll = [pAll;0]; % manually set the last update is time
% run learning method
% do incremental learning.
% randomly choose whether to incremental over time or space
Y_hat = NaN(S, T);
spaceIndex = (1 : length(InitialStations))';

MAE_valid = NaN(1, ValidationSize);
MAE_test = NaN(1, TestingSize);
MAE_test_station = NaN(S,TestingSize);


t = 0;
s = length(spaceIndex);
% randomly initialize the models
A = rand(s, R); B = rand(t, R); C = rand(d, R); V = rand(R,d);
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
        % prepare data
        X_T = squeeze(deszXMonth(spaceIndex, t, :));
        Y_T = squeeze(deszYMonth(spaceIndex, t));
        % update models
        % call update model method for incremental over time
        % preUpdate
        BT = wisdom_incremental_sparsa_time_preUpdate(X_T, A, C, R, lambda, beta);
        % do prediction on X_T
        %             Y_hat_local(spaceIndex,t) = (sum(X_T .* bsxfun(@plus, A * W, BT'*V),2));
        Y_hat(spaceIndex,t) = X_T * (V'*BT);
        MAE_local = nanmean(abs(Y_hat(spaceIndex,t) - Y_T));
        % record the loss
        % if t is in validation period
        if t > TrainingSize && t <= TrainingSize + ValidationSize
            MAE_valid(t-TrainingSize) = MAE_local;
        end
        % if t is in testing period
        if t > TrainingSize + ValidationSize
            MAE_test(t-TrainingSize - ValidationSize) = MAE_local;
            MAE_test_station(spaceIndex, t - TrainingSize - ValidationSize) = abs(Y_hat(spaceIndex,t) - Y_T);
        end
        
        % postUpdate
        [V, A, BT, C] = wisdom_incremental_sparsa_time_postUpdate...
            (X_T, Y_T, V, A, C, lambda, eta, beta, R);
        B = [B; BT'];
    elseif strcmp(tos, 'space')
        % incremental over space
        % incease s
        s = s + 1;
        % prepare data
        newStationIndex = s;
        spaceIndex = [spaceIndex; newStationIndex];
        X_S =reshape( deszXMonth(newStationIndex, 1:t, :), [], d);
        Y_S = reshape(deszYMonth(newStationIndex, 1:t), [], 1);
        % update models
        % call update modele method for incremental over space
        [V, AS, B, C] = wisdom_incremental_sparsa_space...
            (X_S, Y_S, V, B, C, lambda, eta, beta, R);
        A = [A; AS'];
    end
    
end
MAE_ALL = (nanmean(abs(Y_hat - deszYMonth), 2))';
MAE_valid_mean = nanmean(MAE_valid);
MAE_test_mean = nanmean(MAE_test);

save(['WISDOMT-' num2str(responseIndex) '-' num2str(numStartStations) '-' num2str(stationInitScheme) '-' num2str(R) '-' num2str(lambda) '-' num2str(eta) '-' num2str(beta) '-' num2str(randomSeeds) '.mat'], ...
    'MAE_valid', 'MAE_test', 'MAE_ALL', 'MAE_test_station', ...
    'MAE_test_mean', 'MAE_valid_mean', 'V', 'A','B','C', ...
    'Y_hat', 'deszYMonth', ...
    'lambda', 'eta', 'beta', 'R');

