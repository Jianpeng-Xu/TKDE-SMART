% pnopt_stop
% 
%   $Revision: 0.8.0 $  $Date: 2012/12/01 $
%   
    if SWITCH_OPTIM && optim <= optim_tol
      flag    = FLAG_OPTIM;
      %message = MESSAGE_OPTIM;
      message = sprintf('[%u] %s [%.4g:%.4g]', flag, MESSAGE_OPTIM, optim, optim_tol);
      loop    = 0;
    elseif SWITCH_XTOL && norm( x - x_old, 'inf' ) / max( 1, norm( x_old, 'inf' ) ) <= xtol 
      flag    = FLAG_XTOL;
      message = MESSAGE_XTOL;
      loop    = 0;
    elseif SWITCH_FTOL && abs( f_old - f_x ) / max( 1, abs( f_old ) ) <= ftol
      flag    = FLAG_FTOL;
      message = sprintf('[%u] %s [%.4g:%.4g]', flag, MESSAGE_FTOL, abs( f_old - f_x ) / max( 1, abs( f_old ) ), ftol);
      loop    = 0;
    elseif SWITCH_MAXITER && iter >= max_iter 
      flag    = FLAG_MAXITER;
      message = MESSAGE_MAXITER;
      loop    = 0;
    elseif SWITCH_MAXFEV && funEv >= max_fun_evals
      flag    = FLAG_MAXFUNEV;
      % message = fprintf('[%u] %s [%.4g:%.4g]', flag, FLAG_MAXFUNEV, funEv, maxfunEv);
      message = MESSAGE_MAXFUNEV;
      loop    = 0;
    end