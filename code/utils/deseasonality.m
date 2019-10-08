function [desX, seasonCompX ] = deseasonality( X, lengthCycle)
% This function is used to perform deseasonality for the
% timeseries_multivar with cycle length = 12 in default
% output is the deseasoned timeseries, and the season pattern 

[S, T, d] = size(X);
desX = zeros(S, T, d);
% lengthCycle = 12;
seasonCompX = zeros(S, lengthCycle, d);
for s = 1:S
    for i = 1:d
        subdata = squeeze(X(s,:,i));
        % then deseason subdata as a time series
        [desX(s, :, i), seasonCompX(s, :, i)] =  deseasonTS(subdata, lengthCycle);
    end
end

end

