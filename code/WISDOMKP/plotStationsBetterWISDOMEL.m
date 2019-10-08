function plotStationsBetterWISDOMEL(randomSeed)
addpath('../utils/');

% get station locations
dataFile = '../../data/deseasonedMonthly_smallZscore_new.mat';
load(dataFile);
% get stationID, stationLat, stationLon
numStartStations = length(stationID);
stationInitScheme = 1;
% randomSeed = 1;
% use the same random permutation of stations with that in the algorithm
[InitialStations, addStations] = ...
    getStationInitIndex(stationLat, stationLon, numStartStations, stationInitScheme, randomSeed);

stationID = stationID([InitialStations; addStations]);
stationLat = stationLat([InitialStations; addStations]);
stationLon = stationLon([InitialStations; addStations]);

varNames = {'tmax', 'tmin', 'tmean', 'prcp'};

numVar = length(varNames);
buffer = 0.05;

fileNames_wisdom = {'1-100-1-5-200-200-1', ...
    '2-100-1-5-200-250-1',...
    '3-100-1-5-200-350-1',...
    '4-100-1-5-200-0.1-1'};

fileNames_wisdomkp = {'1-100-1-5-100-10-1', ...
    '2-100-1-5-100-1-1',...
    '3-100-1-5-100-100-1',...
    '4-100-1-5-100-100-10'};
num_location_WISDOMPK_outperform_WISDOM = zeros(numVar, 1);
for varIndex = 1 : numVar
    % read the result of WISDOM
    load(['../WISDOM/WISDOM-' fileNames_wisdom{varIndex} '-' num2str(randomSeed) '.mat']);
    % get min_MAE_test_station
    MAE_WISDOM = mean(MAE_test_station, 2);
    
    % read the result of WISDOM-KP
    load(['../WISDOMKP/WISDOMKP-' fileNames_wisdomkp{varIndex} '-' num2str(randomSeed) '.mat']);
    MAE_WISDOM_KP = mean(MAE_test_station, 2);
    
    % First, find the percentage of locations that WISDOMKP outperforms
    % WISDOM
    num_location_WISDOMPK_outperform_WISDOM(varIndex) = nansum(MAE_WISDOM_KP < MAE_WISDOM);
    
    % Plot the stations that WISDOMKP outperform/lowperform WISDOM
    % Get the station index that WISDOM-KP outperforms WISDOM
    outperformStations = find(MAE_WISDOM - buffer >= MAE_WISDOM_KP );
    lowperformStations = find(MAE_WISDOM < MAE_WISDOM_KP - buffer );
%     plotStationLat = stationLat(outperformStations);
%     plotStationLon = stationLon(outperformStations);
    % plot the stations
    h = figure('Name', varNames{varIndex}, 'Position', [100, 100, 600, 300]);
%     box off;
    hold on;
%     worldmap('World');
    land = shaperead('landareas.shp', 'UseGeoCoords', true);
    geoshow(land, 'FaceColor', 'white');
    g1 = geoshow(stationLat(outperformStations), stationLon(outperformStations), 'DisplayType', 'point', ...
        'Marker', '.', 'MarkerSize', 10, 'Color', 'red', 'MarkerEdgeColor', 'auto');
    g2 = geoshow(stationLat(lowperformStations), stationLon(lowperformStations), 'DisplayType', 'point', ...
        'Marker', '.', 'MarkerSize', 10, 'Color', 'blue', 'MarkerEdgeColor', 'auto');
    legend([g1, g2], ...
        {'Stations where WISDOM-KP outperforms WISDOM', 'Stations where WISDOM outperforms WISDOM-KP'},...
        'Location', 'south');
    % save the figure
    axis off;
    set(h,'PaperPositionMode', 'auto');
    print(h, ['OutperformedStationsWorldMap-' varNames{varIndex} '-' num2str(randomSeed)], '-depsc');
    saveas(h, ['OutperformedStationsWorldMap-' varNames{varIndex} '-' num2str(randomSeed) '.fig']);
end

save(['WISDOMPK_vs_WISDOM-' num2str(randomSeed) '.mat'], 'num_location_WISDOMPK_outperform_WISDOM');
    