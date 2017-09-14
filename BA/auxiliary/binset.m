% B = binset(n) Set of n-bit binary numbers
%
% In:
%   n: number of binary variables (bits).
% Out:
%   B: (2^n x n matrix) the 2^n binary numbers in ascending order;
%      each number is 1..n = MSB..LSB.
%
% Any non-mandatory argument can be given the value [] to force it to take
% its default value.

% Copyright (c) 2014 by Miguel A. Carreira-Perpinan

function B = binset(n)

B = logical(dec2bin((0:2^n-1)',n) - '0');

