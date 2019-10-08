function SMARTT(responseIndex, numStartStations, stationInitScheme, randomSeeds, R, lambda, beta)
% This function will use all stations in the training
rng(randomSeeds);
addpath('../utils/');

% responseIndex:
% 1 : tmax; 2: tmin; 3: tmean; 4: precip

% Useful variables:
% deszXMonth: 1118 * 371 * 13
% deszYMonth: 1118 * 371 * 4
dataset = 'deseasonedMonthly_smallZscore_new.mat';
path = '../../data/';
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


	% training models
	[V, A, B, C] = wisdom_sparsa(X_train_valid, Y_train, lambda, beta, R);
	% do prediction on X_valid
	B_valid = B(TrainingSize+1:TrainingSize+ValidationSize, :);
	
	Y_hat = squeeze(sum(X_valid .* repmat(reshape(B_valid*V, [1, ValidationSize, d]), [S, 1, 1]) ,3));
	MAE_valid = mean(abs(Y_hat - Y_valid ), 2);
    

MAE_valid_mean = nanmean(MAE_valid, 2);
% retrain model using best parameters and perform testing on test data 
[V_best, A_best, B_best, C_best] = wisdom_sparsa(X_train_valid_test, Y_train_valid, lambda, beta, R);

% do prediction on X_test
B_test = B_best(TrainingSize+ValidationSize+1:TrainingSize+ValidationSize+TestingSize, :);

Y_hat = squeeze(sum(X_test .* repmat(reshape(B_test*V_best, [1, TestingSize, d]), [S, 1, 1]), 3));
MAE_test = mean(abs(Y_hat - Y_test), 2);
MAE_test_mean = nanmean(MAE_test);

save(['SMARTT-' num2str(responseIndex) '-' num2str(numStartStations) '-' num2str(stationInitScheme) '-' num2str(R) '-' num2str(lambda) '-' num2str(beta) '-' num2str(randomSeeds) '.mat'], ...
    'MAE_test', 'MAE_valid', 'MAE_valid_mean', 'MAE_test_mean', ...
    'V_best', 'A_best', 'B_best', 'C_best', ...
    'Y_hat', 'deszYMonth', ...
    'lambda', 'beta', 'R');

