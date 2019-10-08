function SMARTb_nonIncremental(responseIndex, numStartStations, stationInitScheme, randomSeeds, R, lambda, beta, initModel)
% This function will use all stations in the training
rng(randomSeeds);

% responseIndex:
% 1 : tmax; 2: tmin; 3: tmean; 4: precip

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
deszYMonth = squeeze(deszYMonth([InitialStations; addStations], :, responseIndex)); %1118*371

% create training, validation and testing data 
TrainingSize = 120;
ValidationSize = 120;
TestingSize = T - TrainingSize - ValidationSize;

X_train = deszXMonth(:, 1:TrainingSize, :); % 1118*120*13
Y_train = deszYMonth(:, 1:TrainingSize); % 1118*120

X_valid = deszXMonth(:, TrainingSize+1:TrainingSize+ValidationSize, :); % 1118 * 120 * 13
Y_valid = deszYMonth(:, TrainingSize+1:TrainingSize+ValidationSize); % 1118 * 120

X_test = deszXMonth(:, TrainingSize + ValidationSize + 1 : TrainingSize + ValidationSize + TestingSize, :);
Y_test = deszYMonth(:, TrainingSize + ValidationSize + 1 : TrainingSize + ValidationSize + TestingSize);

X_train_valid = deszXMonth(:, 1:TrainingSize+ValidationSize, :); % 1118*(120+120)*13
Y_train_valid = deszYMonth(:, 1:TrainingSize+ValidationSize); % 1118*(120+120)

X_train_valid_test = deszXMonth(:, 1:TrainingSize+ValidationSize+TestingSize, :); % 1118*(120+120+rest)*13
Y_train_valid_test = deszYMonth(:, 1:TrainingSize+ValidationSize+TestingSize); % 1118*(120+120+rest)


[W, V, A, B, C] = wisdom_sparsa_ModelInit(X_train_valid_test, Y_train_valid, lambda, beta, R, initModel);

% do prediction on X_test
B_test = B(TrainingSize+ValidationSize+1:TrainingSize+ValidationSize+TestingSize, :);

Y_hat = squeeze(sum(X_test .* bsxfun(@plus, reshape(A * W, [S, 1, d]), reshape(B_test*V, [1, TestingSize, d]) ),3));
MAE_test_mean = mean(abs(Y_hat - Y_test), 2);

save(['SMARTb-nonincremental-' num2str(responseIndex) '-' num2str(numStartStations) ...
    '-' num2str(stationInitScheme) '-' num2str(R) '-' num2str(lambda) ...
    '-' num2str(beta) '-' num2str(randomSeeds) '.mat'], ...
    'MAE_test_mean', 'W', 'V', 'A','B','C', ...
    'Y_hat', 'deszYMonth', ...
    'lambda', 'beta', 'R');
