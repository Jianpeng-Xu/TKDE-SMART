function [ x, f_x, output ] = pnopt_accgrad( smoothF, nonsmoothF, x, options )
% pnopt_accgrad : accelerated projected gradient (APG) solver.
%                 The solver uses PNOPT interface.
%
%     NOTE: APG works only for convex problems. 
%     NOTE: For constrained non-smooth acceleration may fail.
%
% Author: Jiayu Zhou, April 20, 2013.


  REVISION = '$Revision: 0.0.1$';
  DATE     = '$Date: April. 01, 2013$';
  REVISION = REVISION(11:end-1);
  DATE     = DATE(8:end-1);
  
% ============ Process options ============
  
  default_options = pnopt_optimset(...
    'debug'         , 0      ,... % debug mode 
    'desc_param'    , 0.0001 ,... % sufficient descent parameter
    'display'       , 100    ,... % display frequency (<= 0 for no display) 
    'backtrack_mem' , 10     ,... % number of previous function values to save
    'maxfunEv'      , 50000  ,... % max number of function evaluations
    'maxIter'       , 5000   ,... % max number of iterations
    'ftol'          , 1e-9   ,... % stopping tolerance on objective function 
    'optim_tol'     , 1e-6   ,... % stopping tolerance on opt
    'xtol'          , 1e-9    ... % stopping tolerance on solution
    );
  
  if nargin > 3
    options = pnopt_optimset( default_options, options );
  else
    options = default_options;
  end
  
  debug         = options.debug;
  display       = options.display;
  maxfunEv      = options.maxfunEv;
  maxIter       = options.maxIter;
  ftol          = options.ftol;
  optim_tol     = options.optim_tol;
  xtol          = options.xtol;
  
% ============ Initialize variables ============
  
  pnopt_flags
  
  iter = 0; 
  loop = 1;
  
  Trace.f_x    = zeros( maxIter + 1, 1 );
  Trace.funEv  = zeros( maxIter + 1, 1 );
  Trace.proxEv = zeros( maxIter + 1, 1 );
  Trace.optim  = zeros( maxIter + 1, 1 );
  
  if debug
    Trace.normDx          = zeros( maxIter, 1 );
    Trace.backtrack_flag  = zeros( maxIter, 1 );
    Trace.backtrack_iters = zeros( maxIter, 1 );
  end
  
  if display > 0    
    fprintf( ' %s\n', repmat( '=', 1, 64 ) );
    fprintf( '        Accelerated Gradient  v.%s (%s)\n', REVISION, DATE );
    fprintf( ' %s\n', repmat( '=', 1, 64 ) );
    fprintf( ' %4s   %6s  %6s  %12s  %12s  %12s \n',...
      '','Fun.', 'Prox', 'Step len.', 'Obj. val.', 'Optim.' );
    fprintf( ' %s\n', repmat( '=', 1, 64 ) );
  end
  
% ------------ Evaluate objective function at starting x ------------
  
  [ f_x, Df_x ] = smoothF(x);
    h_x         = nonsmoothF(x);
    f_x         = f_x + h_x;
  
% ------------ Start collecting data for display and output ------------
  
    funEv       = 1;
    proxEv      = 0;
  [ ~, x_prox ] = nonsmoothF( x - Df_x, 1 );
    optim       = norm( x_prox - x, 'inf' );
  
  Trace.f_x(1)    = f_x;
  Trace.funEv(1)  = funEv;
  Trace.proxEv(1) = proxEv;
  Trace.optim(1)    = optim; 
  
  if display > 0    
    fprintf( ' %4d | %6d  %6d  %12s  %12.4e  %12.4e\n', ...
      iter, funEv, proxEv, '', f_x, optim );
  end
  
% ------------ Check if starting x is optimal ------------
  
  if optim <= optim_tol
    flag    = FLAG_OPTIM;
    message = MESSAGE_OPTIM;
    loop    = 0;
  end

% ============ Main Loop ============
  L = 1;
  alphap=0; alpha=1; xxp=zeros(size(x));

  while loop
    iter = iter + 1; 
    
    beta = (alphap-1)/alpha;  
    s    = x + beta * xxp;  % the new search point 
    
    % function value and gradient. 
    [ f_s, Df_s ] = smoothF(s);
    h_s           = nonsmoothF(s);
    f_s         = f_s + h_s;
    
    x_old = x;
    f_old = f_x;
    
    curvtrack_iters = 0;
    while (1) 
        curvtrack_iters = curvtrack_iters + 1;
        [h_x, x] = nonsmoothF( x - Df_x/L, 1/L );
        v = x - s; % new approximate solution
        [ f_x, Df_x ] = smoothF(x);
        
        r_sum = v' * v;
        l_sum = f_x - f_s - v' * Df_s;
        
        if r_sum <= eps
            curvtrack_flag = 2; % max tolerance
            break; % the gradient step makes little improvement. 
        end
        
        if l_sum < r_sum * L 
            curvtrack_flag = 1; % sufficient decrease
            break;
        else
            L = max(2*L, l_sum/r_sum);
        end
        
    end % end line search
    
    % update alpha and alphap, and check whether converge
    alphap=alpha; alpha= (1+ sqrt(4*alpha*alpha +1))/2;
    xxp= x - x_old;
    
    f_x =f_x + h_x;
    
    step = 1/L; % it seems nto be hessian y. 
    
  % ------------ Collect data and display status ------------
    
      funEv       = funEv + curvtrack_iters;
      proxEv      = proxEv + curvtrack_iters;
    [ ~, x_prox ] = nonsmoothF( x - Df_x ,1);
      optim       = norm( x_prox - x ,'inf');
    
    Trace.f_x(iter+1)    = f_x;
    Trace.funEv(iter+1)  = funEv;
    Trace.proxEv(iter+1) = proxEv;
    Trace.optim(iter+1)  = optim; 
    
    if debug
      Trace.backtrack_flag(iter)  = curvtrack_flag;
      Trace.backtrack_iters(iter) = curvtrack_iters;
    end
    
    if display > 0 && mod( iter, display ) == 0
      fprintf( ' %4d | %6d  %6d  %12.4e  %12.4e  %12.4e\n', ...
        iter, funEv, proxEv, step, f_x, optim );
    end
    
  % ------------ Check stopping criteria ------------
    
    if optim <= optim_tol
      flag    = FLAG_OPTIM;
      message = MESSAGE_OPTIM;
      loop    = 0;
    elseif norm( x - x_old, 'inf' ) / max( 1, norm( x_old, 'inf' ) ) <= xtol 
      flag    = FLAG_XTOL;
      message = MESSAGE_XTOL;
      loop    = 0;
    elseif abs( f_old - f_x ) / max( 1, abs( f_old ) ) <= ftol
      flag    = FLAG_FTOL;
      message = MESSAGE_FTOL;
      loop    = 0;
    elseif iter >= maxIter 
      flag    = FLAG_MAXITER;
      message = MESSAGE_MAXITER;
      loop    = 0;
    elseif funEv >= maxfunEv
      flag    = FLAG_MAXFUNEV;
      message = MESSAGE_MAXFUNEV;
      loop    = 0;
    end
  
  end
  
% ============ Cleanup and exit ============
  
  Trace.f_x    = Trace.f_x(1:iter+1);
  Trace.funEv  = Trace.funEv(1:iter+1);
  Trace.proxEv = Trace.proxEv(1:iter+1);
  Trace.optim  = Trace.optim(1:iter+1);
  
  if debug
    Trace.backtrack_flag  = Trace.backtrack_flag(1:iter);
    Trace.backtrack_iters = Trace.backtrack_iters(1:iter);
  end
  
  if display > 0 && mod(iter,display) > 0
    fprintf( ' %4d | %6d  %6d  %12.4e  %12.4e  %12.4e\n', ...
      iter, funEv, proxEv, step, f_x, optim );
  end
  
  output = struct( ...
    'flag'    , flag    ,...
    'funEv'   , funEv   ,...
    'iters'   , iter    ,...
    'optim'   , optim   ,...
    'options' , options ,...
    'proxEv'  , proxEv  ,...
    'trace'   , Trace    ...
    );
  
  if display > 0
    fprintf( ' %s\n', repmat( '-', 1, 64 ) );
    fprintf( ' %s\n', message )
    fprintf( ' %s\n', repmat( '-', 1, 64 ) );
  end
