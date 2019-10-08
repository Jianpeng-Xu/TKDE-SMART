function correlateTemporalFactorWithResponseVariable
addpath('../utils/');

% get station locations
dataFile = '../../data/deseasonedMonthly_smallZscore_new.mat';
load(dataFile);

% load climate indices
load('../../correlationYAndIndex/desClimateIndicesZscore.mat');
indexNames = {'AOI', 'NAO', 'WPI', 'PDO',  'QBO', 'SOI'};

% load spatial factors
randomSeeds = 0:4;

for randomSeed = randomSeeds
    
    fileNames = {['WISDOM-1-100-1-5-200-200-1-' num2str(randomSeed) '.mat'], ...
        ['WISDOM-2-100-1-5-200-250-1-' num2str(randomSeed) '.mat'],...
        ['WISDOM-3-100-1-5-200-350-1-' num2str(randomSeed) '.mat'],...
        ['WISDOM-4-100-1-5-200-0.1-1-' num2str(randomSeed) '.mat']};
    varNames = {'tmax', 'tmin', 'tmean', 'prcp'};
    
    numVar = length(fileNames);
    corr_TFTV = cell(numVar, 1);
    threshold = 0.3;
    highCorrStationRatio = cell(numVar, 1);
    
    for varIndex = 1 : numVar
        bestModel = load(fileNames{varIndex});
        % get bestModel stucture
        temporalFactor = bestModel.B;
        factors = [temporalFactor, indicesAll'];
        varData = squeeze(deszYMonth(:, :, varIndex))';
        
        corr_local = corr(factors, varData);
        corr_TFTV{varIndex} = corr_local;
        % count the number of stations with high correlation for each factor
        highCorrStationRatio{varIndex} = mean(abs(corr_local) > threshold, 2);
        h = figure('Name', varNames{varIndex}, 'Position', [100, 100, 330, 150]);
        bar(highCorrStationRatio{varIndex});
        ax = gca;
        ax.XTickLabel = {'Factor 1', 'Factor 2', 'Factor 3', 'Factor 4', 'Factor 5', indexNames{:}};
        ax.XTickLabelRotation = 90;
        set(h,'PaperPositionMode', 'auto');
        print(h, ['TemporalFactorVSClimateIndexStationRatio-' varNames{varIndex} '-' num2str(randomSeed) ], '-depsc');
        saveas(h, ['TemporalFactorVSClimateIndexStationRatio-' varNames{varIndex} '-' num2str(randomSeed) '.fig']);
    end
end
