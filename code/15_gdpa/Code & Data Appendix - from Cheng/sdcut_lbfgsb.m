function [u_opt,C_minus_opt,results] = sdcut_lbfgsb(data,opts)
%EQ_10_SOLVE_DUAL_LBFGSB Summary of this function goes here
%   Detailed explanation goes here

% x = lbfgsb( fcn, l, u )
%   uses the lbfgsb v.3.0 library (fortran files must be installed;
%       see compile_mex.m ) which is the L-BFGS-B algorithm.
%   The algorithm is similar to the L-BFGS quasi-Newton algorithm,
%   but also handles bound constraints via an active-set type iteration.
%   This version is based on the modified C code L-BFGS-B-C, and so has 
%   a slightly different calling syntax than previous versions.
%
%  The minimization problem that it solves is:
%       min_x  f(x)     subject to   l <= x <= u
%
% 'fcn' is a function handle that accepts an input, 'x',
%   and returns two outputs, 'f' (function value), and 'g' (function gradient).
%
% 'l' and 'u' are column-vectors of constraints. Set their values to Inf
%   if you want to ignore them. (You can set some values to Inf, but keep
%   others enforced).
%
% The full format of the function is:
% [x,f,info] = lbfgsb( fcn, l, u, opts )
%   where the output 'f' has the value of the function f at the final iterate
%   and 'info' is a structure with useful information
%       (self-explanatory, except for info.err. The first column of info.err
%        is the history of the function values f, and the second column
%        is the history of norm( gradient, Inf ).  )
%
%   The 'opts' structure allows you to pass further options.
%   Possible field name values:
%
%       opts.x0     The starting value (default: all zeros)
%       opts.m      Number of limited-memory vectors to use in the algorithm
%                       Try 3 <= m <= 20. (default: 5 )
%       opts.factr  Tolerance setting (see this source code for more info)
%                       (default: 1e7 ). This is later multiplied by machine epsilon
%       opts.pgtol  Another tolerance setting, relating to norm(gradient,Inf)
%                       (default: 1e-5)
%       opts.maxIts         How many iterations to allow (default: 100)
%       opts.maxTotalIts    How many iterations to allow, including linesearch iterations
%                       (default: 5000)
%       opts.printEvery     How often to display information (default: 1)
%       opts.errFcn         A function handle (or cell array of several function handles)
%                       that computes whatever you want. The output will be printed
%                       to the screen every 'printEvery' iterations. (default: [] )
%                       Results saved in columns 3 and higher of info.err variable

fcn = @(u) sdcut_grad_lbfgsb(u,...
                                          data.A,...
                                          data.B,...
                                          data.b,...
                                          opts.sigma);
lbfgsb_opts=struct('x0',         data.u_init, ...
                   'maxIts',     opts.lbfgsb_maxIts, ...
                   'factr',      opts.lbfgsb_factr, ...
                   'pgtol',      opts.lbfgsb_pgtol, ...
                   'm',          opts.lbfgsb_m,...
                   'printEvery', opts.lbfgsb_printEvery);
t_orig=tic;
[u_opt, obj_lbfgsb, info]=lbfgsb(fcn, data.l_bbox, data.u_bbox, lbfgsb_opts);
t_orig_end=toc(t_orig);

results.iters=info.totalIterations;
results.time=t_orig_end;
results.obj=obj_lbfgsb;

C_minus_opt=sdcut_c_minus(u_opt, data.A, data.B);
end

