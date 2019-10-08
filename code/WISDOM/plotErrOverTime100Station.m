function plotErrOverTime100Station

fileNames = {'WISDOM-1-100-1-5-200-200-1-0.mat', ...
    'WISDOM-2-100-1-5-200-250-1-0.mat', ...
    'WISDOM-3-100-1-5-200-350-1-0.mat', ...
    'WISDOM-4-100-1-5-200-0.1-1-0.mat'};
fileNamesNoIncremental = {'WISDOMNoIncrementalSpace-1-100-1-5-200-200-1-0.mat', ...
    'WISDOMNoIncrementalSpace-2-100-1-5-200-250-1-0.mat', ...
    'WISDOMNoIncrementalSpace-3-100-1-5-200-350-1-0.mat', ...
    'WISDOMNoIncrementalSpace-4-100-1-5-200-0.1-1-0.mat'};

% plotstyle = {'-*r', '-+b', '-vb', '--c', '-sm', '-dy', '-^k', '-vr','->g', '-<b'};


numVar = length(fileNames);
variableNames = {'tmax', 'tmin', 'tmean', 'prcp'};
% MAE_100Station_time = [];
% MAE_100Station_time_noIncrementalSpace = [];

for varIndex = 1 : numVar
    h = figure('Name', variableNames{varIndex}, 'Renderer', 'painters', 'Position', [10 10 400 200]);
    hold on;
    load(fileNames{varIndex});
    % get MAE_ALL_time
    MAE_100Station_time_IncrementalSpace = MAE_100Station_time; 

    load(fileNamesNoIncremental{varIndex});
    % get MAE_ALL_time
    MAE_100Station_time_noIncrementalSpace = MAE_100Station_time;

    % plot MAE_ALL_time
    MAE_100Station_time_year = nanmean( reshape([MAE_100Station_time_IncrementalSpace NaN], [], 12), 2);
    MAE_100Station_time_noIncrementalSpace_year = nanmean( reshape([MAE_100Station_time_noIncrementalSpace NaN], [], 12), 2);    
    year_range = datenum(1985:2015)
    
    plot(year_range, MAE_100Station_time_year, '-r');
    plot(year_range, MAE_100Station_time_noIncrementalSpace_year, '--b');
    hold off;
    xlabel('Year');
    ylabel('MAE');
    legend('With incremental learning over space', 'Without incremental learning over space', 'location', 'north');    
    
    ax = gca;
    ax.XTickLabelRotation = 90;
    xlabel('Year');
    ylabel('MAE');
    
    ylim([0,2.0]);
    % save the figure
    set(h,'PaperPositionMode', 'auto');
    print(h, ['incrementalSpaceTest-' variableNames{varIndex}], '-depsc');
    saveas(h, ['incrementalSpaceTest-' variableNames{varIndex} '.fig']);
end
