% [h,Z] = itq(X,L[,rot]) ITQ and tPCA
%
% Learn binary hash functions with ITQ (iterative quantization) or with tPCA
% (thresholded PCA).
%
% tPCA computes PCA and truncates its low-dim codes using zero as threshold.
% Run it as itq(X,L).
% ITQ computes PCA and rotates its low-dim codes to make them as binary as
% possible, then truncates them. Run it as itq(X,L,0).
%
% In:
%   X: NxD matrix containing N D-dim data points rowwise, the training set.
%   L: number of bits in the Hamming space.
%   rot: 1 to do ITQ rotation, 0 otherwise (ie tPCA). Default: 1.
% Out:
%   h: hash function. It is a struct with parameters:
%      type = 'linh', W = weight matrix (of LxD), w = bias vector (of Lx1).
%   Z: NxL binary matrix, final codes. It equals the output of h(X).

% Any non-mandatory argument can be given the value [] to force it to take
% its default value.

% Copyright (c) 2015 by Ramin Raziperchikolaei and Miguel A. Carreira-Perpinan

function [h,Z] = itq(X,L,rot)

% ---------- Argument defaults ----------
if ~exist('rot','var') || isempty(rot) rot = 1; end;
% ---------- End of "argument defaults" ----------

m = mean(X,1); X = bsxfun(@minus,X,m); U = pca(X,'NumComponents',L);
if rot==1	% ITQ
  ZZ = X*U; [~,R] = ITQ(ZZ,50); U = U*R; Z = (ZZ*R) > 0;
else		% tPCA
  Z = (X*U) > 0;
end
h.type = 'linh'; h.W = U'; h.w = (-m*U)';


function [B,R] = ITQ(V, n_iter)
%
% main function for ITQ which finds a rotation of the PCA embedded data
% Input:
%       V: n*c PCA embedded data, n is the number of images and c is the
%       code length
%       n_iter: max number of iterations, 50 is usually enough
% Output:
%       B: n*c binary matrix
%       R: the c*c rotation matrix found by ITQ
% Author:
%       Yunchao Gong (yunchao@cs.unc.edu)
% Publications:
%       Yunchao Gong and Svetlana Lazebnik. Iterative Quantization: A
%       Procrustes Approach to Learning Binary Codes. In CVPR 2011.
%

% initialize with a orthogonal random rotation
bit = size(V,2);
R = randn(bit,bit);
[U11 S2 V2] = svd(R);
R = U11(:,1:bit);

% ITQ to find optimal rotation
for iter=0:n_iter
    Z = V * R;      
    UX = ones(size(Z,1),size(Z,2)).*-1;
    UX(Z>=0) = 1;
    C = UX' * V;
    [UB,sigma,UA] = svd(C);    
    R = UA * UB';
end

% make B binary
B = UX;
B = B>0;

