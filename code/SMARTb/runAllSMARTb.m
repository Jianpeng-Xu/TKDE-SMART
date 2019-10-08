function runAllSMARTb

responses = 1:4;
num_response = 4;
numStartStations = 100;
stationInitScheme = 1;
randomSeeds = [0 1 2 3 4]; % run 5 times 

lambda = [1 1 1 1];
beta = [0.01 0.01 0.1 0.01];
R = [5 5 5 5];

for i = responses
   for randomSeed = randomSeeds
       fprintf(['SMARTb(' num2str(i) ', ' num2str(numStartStations) ', ' num2str(stationInitScheme) ', ' num2str(randomSeed) ', ' num2str(R(i)) ', ' num2str(lambda(i)) ', ' num2str(beta(i)) ')' '\n']);
       SMARTb(i, numStartStations, stationInitScheme, randomSeed, R(i), lambda(i), beta(i));
   end
end

