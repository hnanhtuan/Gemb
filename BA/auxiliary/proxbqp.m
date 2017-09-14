% [Z,Y,X] = proxbqp(V,m,A,B[,L,U,Z,Y,r,maxit,tol])
% Proximal bound-constrained quadratic program using ADMM
%
% Solves the following strictly convex quadratic program (QP) for each n=1..N:
%   min_xn{ ½xn'.A.xn - bn'.xn + ½m.|xn-vn|² } st xn in [ln,un]
% where:
% - A is of DxD, symmetric positive (semi)definite
% - B = (b1..bN), X = (x1..xN) and V = (v1..vN) are of DxN
% - m >= 0.
%
% Upon convergence, X is an optimal solution and its Lagrange multipliers for
% the auxiliary equality constraint Z = X in ADMM are r*Y, under the
% convention that the multipliers appear with a plus sign in the Lagrangian.
%
% The algorithm uses the alternating direction method of multipliers (ADMM),
% using as penalty parameter r and caching the Cholesky factor of A+r*I.
%
% Notes:
% - If this QP is embedded in an outer loop, one should use as initial values
%   for Z,Y the result from the previous QP's iterate (warm start).
% - We run all n=1:N problems synchronously so the code is vectorised. However,
%   this means that all n run the same number of iterations, which is driven by
%   the slowest zn to converge.
% - If A is (approximately) of rank r, then the penalty parameter r should be
%   sqrt(s1*sr), where s1 >= ... >= sr are the largest r singular values of A.
%   The default currently uses sD instead of sr.
%
% Reference:
% - M. A. Carreira-Perpinan: "An ADMM algorithm for solving a proximal
%   bound-constrained quadratic program".
%   arXiv:1412.8493 [math.OC], Dec. 29, 2014.
%
% In:
%   V: DxN matrix.
%   m: nonnegative scalar.
%   A: DxD matrix.
%   B: DxN matrix. It can be given as a vector of Dx1 (1xN) if all its columns
%      (rows) are equal.
%   L: DxN matrix of lower bounds for x. It can be given as a scalar if all
%      the bounds are equal. Default: 0.
%   U: like L but for upper bounds. Default: 1.
%   Z,Y: DxN matrices, initial values for Z, Y (see output arguments).
%      Default: Z = V, Y = 0.
%   r: positive penalty parameter for ADMM. Default: sqrt(s1*sD) where s1 and
%      sD are the largest and smallest singular values of A.
%   maxit: maximal number of iterations. Default: 1000.
%   tol: small positive number, tolerance in the change of Z to stop iterating.
%      Default: 1e-5.
% Out:
%   Z: DxN matrix, auxiliary variables in ADMM (at the solution, Z = X).
%      If terminating early, so Z ~= X, Z is guaranteed to be in [u,l], so you
%      can use this as an approximate solution that is feasible.
%   Y: DxN matrix, scaled multipliers for the auxiliary constraint in ADMM.
%   X: DxN matrix, solution of the QP.
%
% Any non-mandatory argument can be given the value [] to force it to take
% its default value.

% Copyright (c) 2014 by Miguel A. Carreira-Perpinan

function [Z,Y,X] = proxbqp(V,m,A,B,L,U,Z,Y,r,maxit,tol)

[D,N] = size(V);
% ---------- Argument defaults ----------
if ~exist('L','var') || isempty(L) L = 0; end;
if ~exist('U','var') || isempty(U) U = 1; end;
if ~exist('r','var') || isempty(r) s = svd(A); r = sqrt(s(1)*s(end)); end;
if ~exist('Z','var') || isempty(Z) Z = V; end;
if ~exist('Y','var') || isempty(Y) Y = zeros(D,N); end;
if ~exist('maxit','var') || isempty(maxit) maxit = 1000; end;
if ~exist('tol','var') || isempty(tol) tol = 1e-5; end;
% ---------- End of "argument defaults" ----------

R = chol(A+r*eye(D,D)); Rt = R'; mV = m*V; Zold = zeros(D,N);
for i=1:maxit
  X = R \ (Rt \ bsxfun(@plus,B,r*(Z-Y)));
  Z = (mV+r*(X+Y))/(m+r);
  if isscalar(L) Z(Z<L) = L; else Z(Z<L) = L(Z<L); end
  if isscalar(U) Z(Z>U) = U; else Z(Z>U) = U(Z>U); end
  Y = Y + X - Z;
  if max(abs(Z(:)-Zold(:))) < tol break; end; Zold = Z;
end

