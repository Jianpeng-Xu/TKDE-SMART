%% FUNCTION folding
%  Recover tensor from mold k unfolding matrix Qt of tensor Q  
%  
%% INPUT
%   sz: size of tensor 
%   Qt: mold k unfolding matrix 
%   k : mold k 
%% OUTPUT
%   Q:  tensor
% 
% 
% 
%% Code starts here

function Q = folding(Q,sz,k)
 
nd=length(sz);
sz=sz([k:nd, 1:k-1]);
Q=reshape(Q,sz);           % first convert matrix to tensor of dimension of sz(k)*sz(k+1)..*sz(nd)*sz(1)...sz(k-1)
if k~=1
  Q=permute(Q,[nd-k+2:nd 1:nd-k+1]); % permute the order to original tensor sz(1)*...sz(k)*..*sz(nd)
end


end