function [W, V, A, B, C] = wisdom_sparsa_ModelInit(XT, Y, lambda, beta, R, initModel)

% Input:
% XT: S x (T+1) x d
% Y: S x T

% Output:
% A: S x R
% B: T x R
% C: d x R
% use tensor toolbox

% we use the responseIndex for reading the best model from other algorithms
% in order to give a good start point for this algorithm

addpath('../tensor_toolbox');
addpath('../tensor_toolbox/met/');

rng(0);

[S, T, d] = size(XT);
[~, n] = size(Y);
% initialRnd = 1;
% if initialRnd == 1
%    W = rand(R,d);
%    % V = rand(R,d);
%    V = zeros(R, d);
%    A = rand(S, R);
%    B = rand(T, R);
%    C = rand(d, R);
% end

% read in the best model from SMART-S and set V=0
% 
best_para = initModel;
W = best_para.W_best;
V = zeros(R, d);
A = best_para.A_best(1:S, :);
B = best_para.B_best(1:T, :);
C = best_para.C_best;

%
vect = [A(:); B(:); C(:); W(:); V(:)];
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
V = reshape(vect_result(length(A(:)) + length(B(:))+ length(C(:)) + length(W(:)) + 1 : length(A(:)) + length(B(:)) + length(C(:)) + length(W(:)) + length(V(:))), size(V));
end

function [f, g] = smooth_part(parameterVect, XT, Y, lambda, S, T, n, d, R)
% recover the models from the last iteration

A = reshape(parameterVect(1 : S * R), [S, R]);
B = reshape(parameterVect(length(A(:)) + 1 : length(A(:)) + T*R), [T, R]);
C = reshape(parameterVect(length(A(:)) + length(B(:))+ 1 : length(A(:)) + length(B(:)) + d*R), [d, R]);
W = reshape(parameterVect(length(A(:)) + length(B(:))+ length(C(:)) + 1 : length(A(:)) + length(B(:)) + length(C(:)) + R * d), [R, d]);
V = reshape(parameterVect(length(A(:)) + length(B(:))+ length(C(:)) + length(W(:)) + 1 : length(A(:)) + length(B(:)) + length(C(:)) + length(W(:)) + R * d), [R, d]);

% compute f
XTA = XT(:,1:n,:);

AWBV = bsxfun(@plus, reshape(A*W, [S, 1, d]), reshape(B(1:n,:)*V, [1, n, d]));
AWBVY = sum(XTA .*  AWBV, 3) - Y;
loss = 0.5 * norm(AWBVY, 'fro')^2;

XT_hat = double(ktensor({A, B, C}));

regularizer = 0.5 * lambda * norm(XT(:) - XT_hat(:))^2;
f = loss + regularizer;

% compute gradient
% ======================================================
% A
CKB = kr(C,B);
X1 = double(tenmat(XT, 1));

g_A = reshape(squeeze(sum( bsxfun(@times, XTA, reshape(AWBVY, [S, n, 1]))  ,2)), [S, d]) * W'...
    - lambda * (X1 - A * CKB')*CKB ; 

% B
X2 = double(tenmat(XT, 2));

CKA = kr(C, A);
g_B1 = reshape(squeeze(sum( bsxfun(@times, XTA, reshape(AWBVY, [S, n, 1]))  ,1)), [n, d]) * V';
g_B1(T,R) = 0; % pad the size
g_B = g_B1 - lambda * (X2 - B * CKA') * CKA;


% C
BKA = kr(B, A);
X3 = double(tenmat(XT, 3));

g_C = -lambda * (X3 - C * BKA') * BKA;
% ==================================================================
% W
XTADPAWBVY = bsxfun(@times, XTA, reshape(AWBVY, [S, n, 1]));
g_W = A' * reshape(squeeze(sum(XTADPAWBVY, 2)), [S, d]);

% V
g_V = B(1:n, :)' * reshape(squeeze(sum(XTADPAWBVY, 1)), [n, d]);
% ==================================================================
% Set the gradients to be non-zero if we want to test that gradient
% g_A = zeros(S * R, 1); % good
% g_BT = zeros(R, 1); % good
% g_C = zeros(d * R, 1); % good
% g_W = zeros(R * d, 1); % uncomment to check correctness
% g_V = zeros(R * d, 1); % uncomment to check correctness


g = [g_A(:); g_B(:); g_C(:); g_W(:); g_V(:)];
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
