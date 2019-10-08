function [BT] = wisdom_incremental_sparsa_time_preUpdate(XT, SOI_T, A_old, BT_old, C_old, R, lambda2, beta2)
% Incremental learning over space: add XS with size T x d

% Input:
% XT: S x d

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

vect = BT_old;
%
% function value/gradient of the smooth part
smoothF    = @(parameterVect) smooth_part(parameterVect, XT, lambda2, ...
    R,S, d, A_old, C_old);
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
BT = vect_result;
end

function [f, g] = smooth_part(parameterVect, XT, lambda2, ...
    R, S, d, A_old, C_old)
% recover the models from the last iteration

BT = parameterVect;
% compute f
% ==========================================================
loss = 0;

XT_hat = double(ktensor({A_old, BT', C_old}));
XTT = tensor(reshape(XT, [S 1 d]));

regularizer = 0.5 * lambda2 * norm(XTT(:) - XT_hat(:))^2;
f = loss + regularizer;

% compute gradient
% ======================================================
X2 = double(tenmat(XTT, 2));

CKA = kr(C_old, A_old);
g_BT = - lambda2 * ((X2 - BT' * CKA') * CKA)';

g = g_BT;

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
        x(1) = SOI_T(end);
    end
end
