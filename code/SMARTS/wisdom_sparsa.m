function [W, A, B, C] = wisdom_sparsa(XT, Y, lambda, beta, R)

% This version only consider spatial latent factors in the regression part. 
% Hence, V will not be learned in this function 

% Input:
% XT: S x (T+1) x d
% Y: S x T

% Output:
% A: S x R
% B: T x R
% C: d x R
% use tensor toolbox
addpath('../tensor_toolbox');
addpath('../tensor_toolbox/met/');

rng(0);

[S, T, d] = size(XT);
[~, n] = size(Y);
initialRnd = 1;
if initialRnd == 1
   W = rand(R,d);
   A = rand(S, R);
   B = rand(T, R);
   C = rand(d, R);
end


vect = [A(:); B(:); C(:); W(:)];
%
% function value/gradient of the smooth part
smoothF    = @(parameterVect) smooth_part(parameterVect, XT, Y, ...
    lambda, S, T, n, d, R);
% non-negativen l1 norm proximal operator.
non_smooth = prox_P(beta);
sparsa_options = pnopt_optimset(...
    'display'   , 0    ,...
    'debug'     , 0    ,...
    'maxIter'   , 500  ,...
    'ftol'      , 1e-5 ,...
    'optim_tol' , 1e-5 ,...
    'xtol'      , 1e-5 ...
    );
[vect_result, ~,info] = pnopt_sparsa( smoothF, non_smooth, vect, sparsa_options );

A = reshape(vect_result(1 : length(A(:))), size(A));
B = reshape(vect_result(length(A(:)) + 1 : length(A(:)) + length(B(:))), size(B));
C = reshape(vect_result(length(A(:)) + length(B(:))+ 1 : length(A(:)) + length(B(:)) + length(C(:))), size(C));
W = reshape(vect_result(length(A(:)) + length(B(:))+ length(C(:)) + 1 : length(A(:)) + length(B(:)) + length(C(:)) + length(W(:))), size(W));

end

function [f, g] = smooth_part(parameterVect, XT, Y, lambda, S, T, n, d, R)
% recover the models from the last iteration

A = reshape(parameterVect(1 : S * R), [S, R]);
B = reshape(parameterVect(length(A(:)) + 1 : length(A(:)) + T*R), [T, R]);
C = reshape(parameterVect(length(A(:)) + length(B(:))+ 1 : length(A(:)) + length(B(:)) + d*R), [d, R]);
W = reshape(parameterVect(length(A(:)) + length(B(:))+ length(C(:)) + 1 : length(A(:)) + length(B(:)) + length(C(:)) + R * d), [R, d]);

% compute f
XTA = XT(:,1:n,:);

AW = repmat(reshape(A*W, [S, 1, d]), [1, n, 1]);

AWY = sum(XTA .* AW, 3) - Y;
loss = 0.5 * norm(AWY, 'fro')^2;

XT_hat = double(ktensor({A, B, C}));

regularizer = 0.5 * lambda * norm(XT(:) - XT_hat(:))^2;
f = loss + regularizer;

% compute gradient
% ======================================================
% A
CKB = kr(C,B);
X1 = double(tenmat(XT, 1));

g_A = reshape(squeeze(sum( bsxfun(@times, XTA, reshape(AWY, [S, n, 1]))  ,2)), [S, d]) * W'...
    - lambda * (X1 - A * CKB')*CKB ; 

% B
X2 = double(tenmat(XT, 2));

CKA = kr(C, A);
g_B = - lambda * (X2 - B * CKA') * CKA;


% C
BKA = kr(B, A);
X3 = double(tenmat(XT, 3));

g_C = -lambda * (X3 - C * BKA') * BKA;
% ==================================================================
% W
XTADPAWY = bsxfun(@times, XTA, reshape(AWY, [S, n, 1]));
g_W = A' * reshape(squeeze(sum(XTADPAWY, 2)), [S, d]);

% ==================================================================
% Set the gradients to be non-zero if we want to test that gradient
% g_A = zeros(S * R, 1); % good
% g_BT = zeros(R, 1); % good
% g_C = zeros(d * R, 1); % good
% g_W = zeros(R * d, 1); % uncomment to check correctness
% g_V = zeros(R * d, 1); % uncomment to check correctness


g = [g_A(:); g_B(:); g_C(:); g_W(:)];
end



function op = prox_P(beta) %lambda3,T, d

%PROX_L1    L1 norm.
%    OP = PROX_L1( q ) implements the nonsmooth function
%        OP(X) = norm(q.*X,1).
%    Q is optional; if omitted, Q=1 is assumed. But if Q is supplied,
%    then it must be a positive real scalar (or must be same size as X).
% Dual: proj_linf.m

% Update Feb 2011, allowing q to be a vector
% Update Mar 2012, allow stepsize to be a vector

op = tfocs_prox( @f, @prox_f , 'vector' ); % Allow vector stepsizes
%  lasso for VT
    function v = f(x)
        v = beta * norm(x, 1);
    end

    function x = prox_f(x,t)
        tq = t .* beta; % March 2012, allowing vectorized stepsizes
        s  = 1 - min( tq./abs(x), 1 );
        x  = x .* s;
    end


end
