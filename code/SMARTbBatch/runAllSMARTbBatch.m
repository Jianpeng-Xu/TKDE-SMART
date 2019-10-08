function runAllSMARTbBatch

responses = 1:4;

num_response = 4;
numStartStations = 100;
stationInitScheme = 1;
randomSeeds = [0 1 2 3 4]; % run 5 times 

lambda = [1 1 1 0.01];
beta = [10 0.001 100 10];
R = 5;
d = 13;
S = 1118;
T = 371;

bestModelPara = { ...
    '5-0.001-0.01', ...
    '5-0.001-0.1', ...
    '5-0.01-0.01', ...
    '5-0.001-0.01' ...
    };

for i = responses
   for randomSeed = randomSeeds
		% create initModel for SMARTbBatch
		rng(0);
		initModel.W_best = rand(R,d);
		initModel.V_best = rand(R,d);
		initModel.A_best = rand(S, R);
		initModel.B_best = rand(T, R);
		initModel.C_best = rand(d, R);
		
		fprintf(['SMARTbBatch(' num2str(i) ', ' num2str(numStartStations) ', ' num2str(stationInitScheme) ', ' num2str(randomSeed) ', ' num2str(R) ', ' num2str(lambda(i)) ', ' num2str(beta(i)) ', initModel)' '\n']);
		SMARTbBatch(i, numStartStations, stationInitScheme, randomSeed, R, lambda(i), beta(i), initModel);
   end
end

