function runTime_WISDOMBatch(responseIndex, numStartStations, stationInitScheme)
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

[InitialStations, addStations] = getStationInitIndex(stationLat, stationLon, numStartStations, stationInitScheme, randomSeed);

deszXMonth = deszXMonth([InitialStations; addStations], :, :);
deszYMonth = squeeze(deszYMonth([InitialStations; addStations], :, responseIndex));

TrainingSize = 120;
ValidationSize = 120;
TestingSize = T - TrainingSize - ValidationSize;

R = 5;  lambda = 10; eta = 0.1; beta = 100;

pAll = ones(S+T - numStartStations - 1, 1);
pAll(randperm(length(pAll), T-1)) = 0;
pAll = [pAll;0]; % manually set the last update is time

% run learning method
tic;    % do incremental learning.
    % randomly choose whether to incremental over time or space
    spaceIndex = (1 : length(InitialStations))';
    timeIndex = [];
    % Initialize regularization coefficients
        
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
            [W, V, A, B, C] = wisdom_sparsa(X_T, Y_T, lambda, beta, R);
            % do prediction on X_t only
            X_t = squeeze(X_T(:, t, :));
            Y_t = squeeze(deszYMonth(spaceIndex, timeIndex(t)));
            B_t = squeeze(B(t,:))';
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
runTime = toc;
save(['Runtime-', num2str(responseIndex), '.mat'], 'runTime');