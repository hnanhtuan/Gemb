% [Z rZ] = Zrelaxed(X,f,V,mu,[Z,maxit,tol])
%
% Binary autoencoder Z step: truncated relaxed approximation.
%
% Optimizes over the real codes Z in [0,1] (a convex QP):
%   min_Z{ |X - f(Z)|� + �.|Z - V|� } s.t. 0 >= Z >= 1
% then projects Z onto {0,1} using a greedy truncation procedure where, for
% each bit in sequence 1:L, we pick the value in {0,1} that has smallest
% objective.
%
% In:
%   X: NxD matrix.
%   f: mapping from z to x.
%   V: NxL binary matrix.
%   mu: positive scalar.
%   Z: NxL binary matrix (initial Z). Default: V.
%   maxit: maximal number of iterations. Default: 1000.
%   tol: small positive number, tolerance in the change of Z to stop iterating.
%     Default: 1e-4.
% Out:
%   Z: NxL binary matrix.
%   rZ: NxL matrix containing the relaxed solution before truncation.
%
% Any non-mandatory argument can be given the value [] to force it to take
% its default value.

% Copyright (c) 2015 by Ramin Raziperchikolaei and Miguel A. Carreira-Perpinan

function [Z,rZ] = Zrelaxed(X,f,V,mu,Z,maxit,tol)

% ---------- Argument defaults ----------
if ~exist('Z','var') || isempty(Z) Z = V; end;
if ~exist('maxit','var') || isempty(Z) maxit = 1000; end;
if ~exist('tol','var') || isempty(Z) tol = 1e-4; end;
% ---------- End of "argument defaults" ----------

[N L] = size(V); W = f.W; w = f.w'; X1 = bsxfun(@minus,X,w);

% Solve relaxed QP
rZ = proxbqp(V',mu,W'*W,W'*X1',[],[],Z',[],[],maxit,tol)';

% Sequential greedy truncation
[Q R] = qr(W,0); X = X1*Q;
for n = 1:N
% parfor n = 1:N
  z = rZ(n,:); v = V(n,:); x = X(n,:);
  for i = 1:L
    z(i) = 1; e1 = (norm(x-z*R').^2) + (mu)*(norm(z-v).^2);
    z(i) = 0; e2 = (norm(x-z*R').^2) + (mu)*(norm(z-v).^2);        
    z(i) = e1 < e2;
  end  
  Z(n,:) = z;
end

end

