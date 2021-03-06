function runAllWISDOMWithRunTime
responses = 1:4;

num_response = 4;
numStartStations = 100;
stationInitScheme = 1;
randomSeeds = [0 1 2 3 4];

lambda = [200 200 200 200];
eta = [200 250 350 0.1];
beta = [1 1 1 1];
R = [5 5 5 5];

for i = responses
   for randomSeed = randomSeeds
       fprintf(['runTime_WISDOM(' num2str(i) ', ' num2str(numStartStations) ', ' num2str(stationInitScheme) ', ' num2str(randomSeed) ', ' num2str(R(i)) ', ' num2str(lambda(i)) ', ' num2str(eta(i)) ', ' num2str(beta(i)) ')' '\n']);
       runTime_WISDOM(i, numStartStations, stationInitScheme, randomSeed, R(i), lambda(i), eta(i), beta(i));
   end
end
