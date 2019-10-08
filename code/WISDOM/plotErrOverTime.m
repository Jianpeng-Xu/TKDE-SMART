function plotErrOverTime

fileNames = {'LossOverTimeWISDOM-1-100-1-5-200-200-1-0.mat', ...
    'LossOverTimeWISDOM-2-100-1-5-200-250-1-0.mat', ...
    'LossOverTimeWISDOM-3-100-1-5-200-350-1-0.mat', ...
    'LossOverTimeWISDOM-4-100-1-5-200-0.1-1-0.mat'};

plotstyle = {'-r', '-+g', '-vb', '--c', '-sm', '-dy', '-^k', '-vr','->g', '-<b'};

% figure;
% hold on;

numVar = length(fileNames);

year_range = datenum(1985:2015);

figure('Renderer', 'painters', 'Position', [10 10 400 230])
hold on

for varIndex = 1 : numVar
    load(fileNames{varIndex});
    % get MAE_ALL_time
    % plot MAE_ALL_time
    
    MAE_ALL_time_year = nanmean( reshape([MAE_ALL_time NaN], [], 12), 2);
    fprintf(['varIndex = ' num2str(varIndex)])
    plot(year_range, MAE_ALL_time_year, plotstyle{varIndex});    
    
end

ax = gca;
   
ax.XTickLabelRotation = 90;
xlabel('Year');
ylabel('MAE');
legend('tmax', 'tmin', 'tmean', 'prcp');
