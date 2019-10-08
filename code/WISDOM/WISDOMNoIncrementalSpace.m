function WISDOMNoIncrementalSpace(responseIndex, numStartStations, stationInitScheme, randomSeeds, R, lambda, eta, beta)

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

% run learning method
% do incremental learning.
% randomly choose whether to incremental over time or space
Y_hat = NaN(S, T);
spaceIndex = (1 : length(InitialStations))';

MAE_valid = NaN(1, ValidationSize);
MAE_test = NaN(1, TestingSize);
MAE_test_station = NaN(S,TestingSize);

MAE_All_time = NaN(1, T);
MAE_100Station_time = NaN(1, T);


t = 0;
s = length(spaceIndex);
% randomly initialize the models
rng(0);
A = rand(s, R); B = rand(t, R); C = rand(d, R); W = rand(R,d); V = rand(R,d);

while t < T

        % increase t
        t = t + 1;
        % prepare data
        X_T = squeeze(deszXMonth(spaceIndex, t, :));
        Y_T = squeeze(deszYMonth(spaceIndex, t));
        % update models
        % call update model method for incremental over time
        % preUpdate
        rng(0); BT_old = rand(R,1);
        BT = wisdom_incremental_sparsa_time_preUpdate(X_T, A, BT_old, C, R, lambda, beta);
        % do prediction on X_T
        Y_hat(spaceIndex,t) = (sum(X_T .* bsxfun(@plus, A * W, BT'*V),2));
        MAE_local = mean(abs(Y_hat(spaceIndex,t) - Y_T));
        MAE_All_time(t) = MAE_local;
        MAE_100Station_time(t) = mean(abs(Y_hat(1:100,t) - Y_T(1:100)));
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
        [W, V, A, BT, C] = wisdom_incremental_sparsa_time_postUpdate...
            (X_T, Y_T, W, V, A, BT_old, C, lambda, eta, beta, R);
        B = [B; BT'];  
end

save(['WISDOMNoIncrementalSpace-' num2str(responseIndex) '-' num2str(numStartStations) '-' num2str(stationInitScheme) '-' num2str(R) '-' num2str(lambda) '-' num2str(eta) '-' num2str(beta) '-' num2str(randomSeeds) '.mat'], ...
    'MAE_valid', 'MAE_test', 'MAE_All_time', 'MAE_100Station_time', 'MAE_test_station', ...
    'W', 'V', 'A','B','C', ...
    'Y_hat', 'deszYMonth', ...
    'lambda', 'eta', 'beta', 'R');

