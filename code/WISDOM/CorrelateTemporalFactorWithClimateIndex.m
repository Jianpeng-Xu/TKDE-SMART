function CorrelateTemporalFactorWithClimateIndex

fileNames = {'WISDOM-1-100-1-5-200-200-1-0.mat', ...
    'WISDOM-2-100-1-5-200-250-1-0.mat',...
    'WISDOM-3-100-1-5-200-350-1-0.mat',...
    'WISDOM-4-100-1-5-200-0.1-1-0.mat'};
varNames = {'tmax', 'tmin', 'tmean', 'prcp'};

numVar = length(fileNames);

load('../../data/desClimateIndicesZscore.mat');
% get indicesAll
indicesAll = indicesAll';
numIndex = size(indicesAll, 2);
RHO = cell(numVar, 1);

for varIndex = 1 : numVar
    bestModel = load(fileNames{varIndex});
    temporalFactor = bestModel.B;
    numFactor = size(temporalFactor, 2);
    RHO_local = abs(corr(temporalFactor, indicesAll));
    RHO{varIndex} = RHO_local;
    % plot the heatmap for each variable
    h = figure('Name', varNames{varIndex}, 'Position', [100, 100, 300, 120]);
    imagesc(RHO_local);
    %    colormap(contrast(RHO_local));
    ch = colorbar;
    ch.Limits = [0 0.6];
    ch.Ticks = linspace(0, 0.6, 4);
%     ch.TickLabels = num2cell(0:0.2:0.6);
    ax = gca;
    ax.XTick = 1 : numIndex;
    ax.XTickLabel = {'AOI', 'NAO', 'WPI', 'PDO', 'QBO', 'SOI'};
    ax.XTickLabelRotation = 90;
    xlim([0 numIndex]);
    ax.YTick =  1 : numFactor;
%     ylim([0 numFactor])
    %    caxis([-0.6 0.6]);
    % save the figure
    set(h,'PaperPositionMode', 'auto');
    print(h, ['TemporalFactorVSClimateIndex-' varNames{varIndex}], '-depsc');
    saveas(h, ['TemporalFactorVSClimateIndex-' varNames{varIndex} '.fig']);
end

