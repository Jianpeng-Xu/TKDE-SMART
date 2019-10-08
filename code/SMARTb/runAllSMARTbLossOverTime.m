function runAllSMARTbLossOverTime

responses=1：4
num_response = 4;
numStartStations = 100;
stationInitScheme = 1;

randomSeeds = 0;
lambda = [1 1 1 1];
beta = [0.01 0.01 0.1 0.01];
R = [5 5 5 5];

for i = responses
   for randomSeed = randomSeeds
       fprintf(['LossOverTimeSMARTb(' num2str(i) ', ' num2str(numStartStations) ', ' num2str(stationInitScheme) ', ' num2str(randomSeed) ', ' num2str(R(i)) ', ' num2str(lambda(i)) ', ' num2str(beta(i)) ')' '\n']);
       LossOverTimeSMARTb(i, numStartStations, stationInitScheme, randomSeed, R(i), lambda(i), beta(i));
   end
end

