function runAllWISDOMKP

responses = 1:4;

num_response = 4;
numStartStations = 100;
stationInitScheme = 1;
randomSeeds = [0 1 2 3 4]; % run 5 times 

lambda = [100 100 100 100];
eta = [10 1 100 100];
beta = [1 1 1 10];
R = [5 5 5 5];

for i = responses
   for randomSeed = randomSeeds
       fprintf(['WISDOMKP(' num2str(i) ', ' num2str(numStartStations) ', ' num2str(stationInitScheme) ', ' num2str(randomSeed) ', ' num2str(R(i)) ', ' num2str(lambda(i)) ', ' num2str(eta(i)) ', ' num2str(beta(i)) ')' '\n']);
       WISDOMKP(i, numStartStations, stationInitScheme, randomSeed, R(i), lambda(i), eta(i), beta(i));
   end
end

