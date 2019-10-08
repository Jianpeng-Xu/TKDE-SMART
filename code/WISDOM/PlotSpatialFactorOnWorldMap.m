function PlotSpatialFactorOnWorldMap

% get station locations
dataFile = '../../data/deseasonedMonthly_smallZscore_new.mat';
addpath('../utils/');

load(dataFile);
% get stationID, stationLat, stationLon
numStartStations = length(stationID);
stationInitScheme = 1;
% use the same random permutation of stations with that in the algorithm
randomSeed = 0;
[InitialStations, addStations] = ...
    getStationInitIndex(stationLat, stationLon, numStartStations, stationInitScheme, randomSeed);

stationID = stationID([InitialStations; addStations]);
stationLat = stationLat([InitialStations; addStations]);
stationLon = stationLon([InitialStations; addStations]);
numStation = length(stationID);

fileNames = {'WISDOM-1-100-1-5-200-200-1-0.mat', ...
    'WISDOM-2-100-1-5-200-250-1-0.mat',...
    'WISDOM-3-100-1-5-200-350-1-0.mat',...
    'WISDOM-4-100-1-5-200-0.1-1-0.mat'};
varNames = {'tmax', 'tmin', 'tmean', 'prcp'};

numVar = length(fileNames);
% descretizationBin = 5;
% load color
% load('mycolor.mat');
myColor = jet;
myColor = myColor(1:7:end, :);
for varIndex = 1 : numVar
    rng(0);
    load(fileNames{varIndex});
    % get bestModel stucture
    spatioFactor = A;
    % plot the first component
    for factorIndex = 1 : size(spatioFactor, 2)
        h = figure('Name', [varNames{varIndex} '-' num2str(factorIndex)], 'Position', [100, 100, 600, 300]);
        land = shaperead('landareas.shp', 'UseGeoCoords', true);
        geoshow(land, 'FaceColor', 'white');
        hold on;
        % rescale spatioFactor
        spatioFactorLocal = spatioFactor(:, factorIndex);
        maxFactor = max(spatioFactorLocal);
        minFactor = min(spatioFactorLocal);
        spatioFactorLocal = (spatioFactorLocal - minFactor)/(maxFactor - minFactor);
        
        for stationIndex = 1 : numStation
            spatioFactorValue = spatioFactorLocal(stationIndex);
            colorIndex = ceil(spatioFactorValue/0.1 + 0.00001);
            if colorIndex > 10 colorIndex = 10; end
            geoshow(stationLat(stationIndex), stationLon(stationIndex), ...
                'DisplayType', 'point', 'Marker', '.', 'MarkerSize', 8, ...
                'Color', myColor(colorIndex,:), 'MarkerEdgeColor', 'auto');
        end
        colormap(jet);
        colorbar;
        % save the figure
        axis off;
        set(h,'PaperPositionMode', 'auto');
        print(h, ['spatialFactor-' num2str(factorIndex) '-WorldMap-' varNames{varIndex}], '-depsc');
        saveas(h, ['spatialFactor-' num2str(factorIndex) '-WorldMap-' varNames{varIndex} '.fig']);        
    end
end