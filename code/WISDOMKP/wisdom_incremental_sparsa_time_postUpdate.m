function [W, V, A, BT, C] = wisdom_incremental_sparsa_time_postUpdate(XT, Y, SOI_T, W_old, V_old, A_old, BT_old, C_old, lambda2, eta2, beta2, R)
% Incremental learning over space: add XS with size T x d

% Input:
% XT: S x d
% Y: S x 1
% BT_old should be a random vector.

% Output:
% A: S x R
% B: T x R
% C: d x R
% use tensor toolbox
addpath('../tensor_toolbox');
addpath('../tensor_toolbox/met/');

% constraint the first component to be SOI
BT_old(1) = SOI_T(end);

[S, d] = size(XT);

vect = [A_old(:); BT_old(:); C_old(:); W_old(:); V_old(:)];
%
% function value/gradient of the smooth part
smoothF    = @(parameterVect) smooth_part(parameterVect, XT, Y, ...
    lambda2, eta2, R,S, d, W_old, V_old, A_old, C_old);
% non-negativen l1 norm proximal operator.
non_smooth = prox_L1(SOI_T, R, S, d, beta2);
sparsa_options = pnopt_optimset(...
    'display'   , 0    ,...
    'debug'     , 0    ,...
    'maxIter'   , 10  ,...
        'maxfunEv'      , 50  ,... % max number of function evaluations
    'ftol'      , 1e-5 ,...
    'optim_tol' , 1e-5 ,...
    'xtol'      , 1e-5 ...
    );
[vect_result, ~,info] = pnopt_sparsa( smoothF, non_smooth, vect, sparsa_options );

A = reshape(vect_result(1 : length(A_old(:))), size(A_old));
BT = reshape(vect_result(length(A_old(:)) + 1 : length(A_old(:)) + length(BT_old(:))), size(BT_old));
C = reshape(vect_result(length(A_old(:)) + length(BT_old(:))+ 1 : length(A_old(:)) + length(BT_old(:)) + length(C_old(:))), size(C_old));
W = reshape(vect_result(length(A_old(:)) + length(BT_old(:))+ length(C_old(:)) + 1 : ...
    length(A_old(:)) + length(BT_old(:)) + length(C_old(:)) + length(W_old(:))), size(W_old));
V = reshape(vect_result(length(A_old(:)) + length(BT_old(:))+ length(C_old(:)) + length(W_old(:)) + 1 : ...
    length(A_old(:)) + length(BT_old(:)) + length(C_old(:)) + length(W_old(:)) + length(V_old(:))), size(V_old));
end

function [f, g] = smooth_part(parameterVect, XT, Y, ...
    lambda2, eta2, R, S, d, W_old, V_old, A_old, C_old)
% recover the models from the last iteration

A = reshape(parameterVect(1 : S*R), [S, R]);
BT = reshape(parameterVect(length(A(:)) + 1 : ...
    length(A(:)) + R), [R, 1]);
C = reshape(parameterVect(length(A(:)) + length(BT(:))+ 1 : ...
    length(A(:)) + length(BT(:)) + d*R), [d, R]);
W = reshape(parameterVect(length(A(:)) + length(BT(:))+ length(C(:)) + 1 :  ...
    length(A(:)) + length(BT(:)) + length(C(:)) + R * d), [R, d]);
V = reshape(parameterVect(length(A(:)) + length(BT(:))+ length(C(:)) + length(W(:)) + 1 : ...
    length(A(:)) + length(BT(:)) + length(C(:)) + length(W(:)) + R * d), [R, d]);

% compute f
% ==========================================================
% loss fast
% loss = 0.5 * norm(sum(XS .* bsxfun(@plus, AS' * W, B'*V),2) - Y)^2;
sloss = sum(XT .* bsxfun(@plus, A * W, BT'*V),2) - Y; 
loss = 0.5 * norm(sloss)^2;

XT_hat = double(ktensor({A, BT', C}));

XTT = tensor(reshape(XT, [S, 1, d]));

regularizer = 0.5 * lambda2 * norm(XTT(:) - XT_hat(:))^2 + ...
    0.5 * eta2 * (norm(W-W_old, 'fro')^2 + norm(V - V_old, 'fro')^2 + ...
    norm(A - A_old, 'fro')^2 + norm(C - C_old, 'fro')^2  );
f = loss + regularizer;


% compute gradient
% ======================================================
% A
CKB = kr(C,BT');
X1 = double(tenmat(XTT, 1));

g_A = bsxfun(@times, sloss, XT*W') - lambda2 * (X1 - A * CKB')*CKB + eta2 * (A - A_old); 

% BT
X2 = double(tenmat(XTT, 2));

CKA = kr(C, A);
g_BT = V * XT' * sloss - lambda2 * ((X2 - BT' * CKA') * CKA)' ;

% C
BKA = kr(BT', A);
X3 = double(tenmat(XTT, 3));

g_C = -lambda2 * (X3 - C * BKA') * BKA + eta2 * (C - C_old);
% ==================================================================
% W
g_W = A' * bsxfun(@times, XT, sloss) + eta2 * (W - W_old);
% V
g_V = BT * sum(bsxfun(@times, XT, sloss), 1) + eta2 * (V - V_old);
% ==================================================================
% Set the gradients to be non-zero if we want to test that gradient
% g_A = zeros(S * R, 1); % good
% g_BT = zeros(R, 1); % good
% g_C = zeros(d * R, 1); % good
% g_W = zeros(R * d, 1); % uncomment to check correctness
% g_V = zeros(R * d, 1); % uncomment to check correctness


g = [g_A(:); g_BT(:); g_C(:); g_W(:); g_V(:)];

end




function op = prox_L1(SOI_T, R, S, d, beta2) 

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
        v = beta2 * norm(x,1);
    end

    function x = prox_f(x,t)
        tq = t .* beta2; % March 2012, allowing vectorized stepsizes
        s  = 1 - min( tq./abs(x), 1 );
        x  = x .* s;
        x(S*R + 1) = SOI_T(end);
        
    end
end
