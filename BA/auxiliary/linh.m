% Y = linh(X,h)
%
% Value of step linear function y = h(x) = step(W.x+w), where
%   step(t) = 1 if t>0, 0 otherwise
% applies elementwise.
%
% In:
%   X: NxL matrix, N L-dim data points rowwise.
%   h: (struct) hash function (containing D binary functions).
% Out:
%   Y: NxD logical matrix, N D-dim outputs Y = h(X).

% Copyright (c) 2015 by Ramin Raziperchikolaei and Miguel A. Carreira-Perpinan

function Y = linh(X,h)

Y = linf(X,h) > 0;

