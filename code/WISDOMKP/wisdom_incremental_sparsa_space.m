function [W, V, AS, B, C] = wisdom_incremental_sparsa_space(XS, Y, SOI_S, W_old, V_old, AS_old, B_old, C_old, lambda1, eta1, beta1, R)
% Incremental learning over space: add XS with size T x d

% Input:
% XS: T x d
% Y: T x 1
% As_old should be a random vector.

% Output:
% A: S x R
% B: T x R
% C: d x R
% use tensor toolbox
addpath('../tensor_toolbox');
addpath('../tensor_toolbox/met/');

[T, d] = size(XS);

vect = [AS_old(:); B_old(:); C_old(:); W_old(:); V_old(:)];
%
% function value/gradient of the smooth part
smoothF    = @(parameterVect) smooth_part(parameterVect, XS, Y, ...
    lambda1, eta1, R,T, d, W_old, V_old, B_old, C_old);
% non-negativen l1 norm proximal operator.
non_smooth = prox_L1(SOI_S, R, T, d, beta1);
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

AS = reshape(vect_result(1 : length(AS_old(:))), size(AS_old));
B = reshape(vect_result(length(AS_old(:)) + 1 : length(AS_old(:)) + length(B_old(:))), size(B_old));
C = reshape(vect_result(length(AS_old(:)) + length(B_old(:))+ 1 : length(AS_old(:)) + length(B_old(:)) + length(C_old(:))), size(C_old));
W = reshape(vect_result(length(AS_old(:)) + length(B_old(:))+ length(C_old(:)) + 1 : ...
    length(AS_old(:)) + length(B_old(:)) + length(C_old(:)) + length(W_old(:))), size(W_old));
V = reshape(vect_result(length(AS_old(:)) + length(B_old(:))+ length(C_old(:)) + length(W_old(:)) + 1 : ...
    length(AS_old(:)) + length(B_old(:)) + length(C_old(:)) + length(W_old(:)) + length(V_old(:))), size(V_old));
end

function [f, g] = smooth_part(parameterVect, XS, Y, ...
    lambda1, eta1, R, T, d, W_old, V_old, B_old, C_old)
% recover the models from the last iteration

AS = parameterVect(1 : R);
B = reshape(parameterVect(length(AS(:)) + 1 : ...
    length(AS(:)) + T*R), [T, R]);
C = reshape(parameterVect(length(AS(:)) + length(B(:))+ 1 : ...
    length(AS(:)) + length(B(:)) + d*R), [d, R]);
W = reshape(parameterVect(length(AS(:)) + length(B(:))+ length(C(:)) + 1 :  ...
    length(AS(:)) + length(B(:)) + length(C(:)) + R * d), [R, d]);
V = reshape(parameterVect(length(AS(:)) + length(B(:))+ length(C(:)) + length(W(:)) + 1 : ...
    length(AS(:)) + length(B(:)) + length(C(:)) + length(W(:)) + R * d), [R, d]);

% compute f
% ==========================================================
% loss fast
% loss = 0.5 * norm(sum(XS .* bsxfun(@plus, AS' * W, B'*V),2) - Y)^2;
tloss = sum(XS .* bsxfun(@plus, AS' * W, B*V),2) - Y; 
loss = 0.5 * norm(tloss)^2;

% loss slow
% loss = 0;
% for t = 1:T
%     loss = loss + 0.5 * norm(XS(t,:) * (W' * AS + V' * B(t,:)') - Y(t))^2;
% end

XT_hat = double(ktensor({AS', B, C}));
% XT_hat = squeeze(double(XT_hat)); 
XST = tensor(reshape(XS, [1 T d]));

regularizer = 0.5 * lambda1 * norm(XST(:) - XT_hat(:))^2 + ...
    0.5 * eta1 * (norm(W-W_old, 'fro')^2 + norm(V - V_old, 'fro')^2 + ...
    norm(B - B_old, 'fro')^2 + norm(C - C_old, 'fro')^2  );
f = loss + regularizer;


% compute gradient
% ======================================================
% A(S+1)
CKB = kr(C,B);
X1 = double(tenmat(XST, 1));

g_AS = W * XS' * tloss - lambda1 * ((X1 - AS' * CKB')*CKB)'; 
% g_AS = W * XS' * tloss; 

% B
X2 = double(tenmat(XST, 2));

CKA = kr(C, AS');
g_B = bsxfun(@times, tloss, XS * V') - lambda1 * ((X2 - B * CKA') * CKA) + eta1 * (B - B_old);

% C
BKA = kr(B, AS');
X3 = double(tenmat(XST, 3));

g_C = -lambda1 * (X3 - C * BKA') * BKA + eta1 * (C - C_old);
% ==================================================================
% W
g_W = AS * sum(bsxfun(@times, XS, tloss), 1) + eta1 * (W - W_old);

% V
g_V = B' * bsxfun(@times, XS, tloss) + eta1 * (V - V_old);
% ==================================================================
% Set the gradients to be non-zero if we want to test that gradient
% g_AS = zeros(R, 1); % good
% g_B = zeros(T * R, 1); % good
% g_C = zeros(d * R, 1); % good
% g_W = zeros(R * d, 1); % uncomment to check correctness
% g_V = zeros(R * d, 1); % uncomment to check correctness


g = [g_AS(:); g_B(:); g_C(:); g_W(:); g_V(:)];

end




function op = prox_L1(SOI_S, R, T, d, beta1) 

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
        v = beta1 * norm(x,1);
    end

    function x = prox_f(x,t)
        tq = t .* beta1; % March 2012, allowing vectorized stepsizes
        s  = 1 - min( tq./abs(x), 1 );
        x  = x .* s;
        B = reshape(x(R + 1 : R + T*R), [T, R]);
        B(:,1) = SOI_S;
    end
end
