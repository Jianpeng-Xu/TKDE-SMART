% pnopt_flags 
% 
%   $Revision: 0.8.0 $  $Date: 2014/10/01 $
% 

  SWITCH_OPTIM   = true;
  SWITCH_XTOL    = true;
  SWITCH_FTOL    = true;
  SWITCH_MAXITER = true;
  SWITCH_MAXFEV  = true;
  SWITCH_OTHER   = true;

  FLAG_OPTIM    = 1;
  FLAG_XTOL     = 2;
  FLAG_FTOL     = 3;
  FLAG_MAXITER  = 4;
  FLAG_MAXFUNEV = 5;
  FLAG_MAXFEV   = 5;
  FLAG_OTHER    = 6;
  
  MESSAGE_OPTIM   = 'Optimality below optim_tol.';
  MESSAGE_XTOL    = 'Relative change in x below xtol.';
  MESSAGE_FTOL    = 'Relative change in function value below ftol.';
  MESSAGE_MAXITER = 'Max number of iterations reached.';
  MESSAGE_MAXFUNEV  = 'Max number of function evaluations reached.';
  MESSAGE_MAXFEV  = 'Max number of function evaluations reached.';
