% [f,fX,E] = linftrain(X,Y[,l]) Train linear function y = f(x) = W.x+w
%
% In:
%   X: NxL matrix, N L-dim data points rowwise.
%   Y: NxD matrix, N D-dim data points rowwise.
%   l: (nonnegative scalar) regularisation parameter. Default: 0.
% Out:
%   f: (struct) the linear function, with fields:
%      type='linf', W (DxL), w (Dx1), regularisation parameter l.
%   fX: NxD matrix, f(X).
%   E: 1x2 list, fit error and regularisation error.

% Copyright (c) 2010 by Miguel A. Carreira-Perpinan

function [f,fX,E] = linftrain(X,Y,l)

% ---------- Argument defaults ----------
if ~exist('l','var') l = []; f.l = 0; else f.l = l; end;
% ---------- End of "argument defaults" ----------

[N,L] = size(X);
f.type = 'linf';

X1 = sum(X,1)'; XX = X'*X - X1*(X1'/N);
if ~isempty(l) XX = XX + spdiags(repmat(l,L,1),0,L,L); end
f.W = (Y'*X-mean(Y,1)'*X1') / XX; f.w = mean(Y-X*f.W',1)';

if nargout>1 fX = linf(X,f); end
if nargout>2
  E = sum(sum((Y-fX).^2));					% Fit error
  %if ~isempty(l) E = [E l*(f.W(:)'*f.W(:))]; else E=[E 0]; end	% Regul. error
end

