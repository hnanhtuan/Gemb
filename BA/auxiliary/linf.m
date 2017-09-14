% [Y,J] = linf(X,f) Value of linear function y = f(x) = W.x+w
%
% See linftrain.
%
% In:
%   X: NxL matrix, N L-dim data points rowwise.
%   f: (struct) the linear function.
% Out:
%   Y: NxD matrix, N D-dim outputs Y = f(X).
%   J: DxL Jacobian matrix (assumes N=1 input only).

% Copyright (c) 2009 by Miguel A. Carreira-Perpinan

function [Y,J] = linf(X,f)
Y = bsxfun(@plus,X*f.W',f.w');
if nargout>1 J = f.W; end

