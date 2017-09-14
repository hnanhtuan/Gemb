function [B, U, Ub] = compressITQ(X, ITQparam)

% Input:
%          X: n*d data matrix, n is number of images, d is dimension
%          ITQparam:
%                           ITQparam.pcaW---PCA of all the database
%                           ITQparam.nbits---encoding length
%                           ITQparam.r---ITQ rotation projection
% output:
%            B: compacted binary code
%            U: binary code
if (isfield(ITQparam, 'pcaW') == 1)
    V = X*ITQparam.pcaW;
else
    V = X;
end

% rotate the data
U = V*ITQparam.r;

B = compactbit(U>0);
Ub = (U>0);



