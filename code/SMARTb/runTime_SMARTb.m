function runTime_SMARTb(responseIndex, numStartStations, stationInitScheme, randomSeeds, R, lambda, beta)
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[S, T, d] = size(deszXMonth);
deszYMonth = deszYMonth(1:S, 1:T, :);

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
timeIndex = [];

MAE_valid = NaN(1, ValidationSize);
MAE_test = NaN(1, TestingSize);
MAE_test_station = NaN(S,TestingSize);


t = 0;
s = length(spaceIndex);

pIndex = 1;
tic;
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
        [W, V, A, B, C] = wisdom_sparsa(X_T, Y_T, lambda, beta, R);
        % do prediction on X_t only
        X_t = squeeze(X_T(:, t, :));
        Y_t = squeeze(deszYMonth(spaceIndex, timeIndex(t)));
        B_t = squeeze(B(t,:))';
        Y_hat(spaceIndex,t) = (sum(X_t .* bsxfun(@plus, A * W, B_t'*V),2));
        MAE_local = mean(abs(Y_hat(spaceIndex,t) - Y_t));
        % record the loss
        % if t is in validation period
        if t > TrainingSize && t <= TrainingSize + ValidationSize
            MAE_valid(t-TrainingSize) = MAE_local;
        end
        % if t is in testing period
        if t > TrainingSize + ValidationSize
            MAE_test(t-TrainingSize - ValidationSize) = MAE_local;
            MAE_test_station(spaceIndex, t - TrainingSize - ValidationSize) = abs(Y_hat(spaceIndex,t) - Y_t);
        end
        fprintf([num2str(t), ' ']);
    elseif strcmp(tos, 'space')
        % incremental over space
        % incease s
        s = s + 1;
        % update space index
        newStationIndex = s;
        spaceIndex = [spaceIndex; newStationIndex];
        % do nothing: no test is performed with only new station
    end
    
end
runTime = toc;
% compute MAE
MAE_ALL = (nanmean(abs(Y_hat - deszYMonth), 2))';
MAE_valid_mean = nanmean(MAE_valid);
MAE_test_mean = nanmean(MAE_test);

save(['SMARTb-withRunTime-' num2str(responseIndex) '-' num2str(numStartStations) '-' num2str(stationInitScheme)  '-' num2str(R) '-' num2str(lambda) '-' num2str(beta) '-' num2str(randomSeeds) '.mat'], ...
    'MAE_valid', 'MAE_test', 'MAE_ALL', 'MAE_test_station', ...
    'MAE_test_mean', 'MAE_valid_mean', 'W', 'V', 'A','B','C', ...
    'Y_hat', 'deszYMonth', ...
    'lambda', 'beta', 'R', 'runTime');
