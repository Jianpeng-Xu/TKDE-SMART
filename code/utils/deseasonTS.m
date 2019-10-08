function [desTS, seasonComp ]= deseasonTS(y, lengthCycle)
% y is a time series, d is the length of one cycle
% desTS is the deseasoned output time series
% In this function, only the seasonality is removed, not the trend
% T = length(y);
% numCycle =floor(T/lengthCycle);
% maty = reshape(y, [lengthCycle, numCycle]);
% maty = maty'; % cxd
% 
% seasonComp = 1/numCycle * sum(maty); % 1xd
% desTS = y - repmat(seasonComp, [1, numCycle]);

seasonComp = zeros(1, lengthCycle);
for i = 1 : lengthCycle
    seasonComp(i) = nanmean(y(i : lengthCycle:end));
    y(i : lengthCycle : end) = y(i : lengthCycle : end) - seasonComp(i);
end
desTS = y;
end