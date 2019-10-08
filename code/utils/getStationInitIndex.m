function [InitializedStationIndex, addStationIndex] = getStationInitIndex(stationLat, stationLon, numStartStations, initScheme, randomSeeds)
% This function is used for selecting the initial stations and the order of
% adding new stations. 
rng(randomSeeds);
numStation = length(stationLat);
if initScheme == 1 % random initialization
    p = randperm(numStation);
    InitializedStationIndex = p(1:numStartStations);
    addStationIndex = p(numStartStations + 1 : end);
elseif initScheme == 2 % cluster centroid initialization
    [~, ~, ~, ~, idx] = kmedoids([stationLat, stationLon], numStartStations);
    InitializedStationIndex = idx; 
    addStationIndex = setdiff(1:numStation, idx);
    p = randperm(length(addStationIndex));
    addStationIndex = addStationIndex(p);
end

InitializedStationIndex = InitializedStationIndex(:);
addStationIndex = addStationIndex(:);