% [h,Z,f] = ba(X,L,mu,[Z,V,enum]) Binary Autoencoder (BA)
%
% Train a binary autoencoder using a MAC algorithm. The encoder can be
% used as a binary hash function for information retrieval.
%
% Notes:
% - We use a validation set V to check the precision at each step. By default,
%   this is a random subset of 200 points from X, and we do not remove them
%   from the training set. If you want disjoint training and validation sets,
%   provide them explicitly as arguments.
% - Internally, BA.m normalizes the data points to [0,1] and undoes this
%   normalization before returning the final hash function (so the process is
%   transparent to the user). This simplifies setting set the schedule for the
%   mu parameter, and makes LIBLINEAR train the SVMs more efficiently.
%
% In:
%   X: NxD matrix containing N D-dim data points rowwise, the training set.
%   L: number of bits in the Hamming space (= number of binary hash functions).
%   mu: list of penalty parameter values; we run one MAC iteration for each.
%   Z: NxL binary matrix containing N L-bit binary data points (0/1) rowwise
%      (initial codes). Default: ITQ.
%   V: MxD matrix containing M D-dim data points rowwise, the validation set.
%      We skip a value of mu if the precision on V does not increase.
%      Default: 200 points randomly selected from dataset X.
%   enum: if L <= enum, use enumeration (exact) optimization in the Z-step,
%      otherwise use alternating optimization. Default: 10.
% Out:
%   h: hash function (encoder). It is a struct with parameters:
%      type = 'linh', W = weight matrix (of LxD), w = bias vector (of Lx1).
%   Z: NxL binary matrix, final codes. It equals the output of h(X).
%   f: decoder function. It is a struct with parameters:
%      type = 'linf', W = weight matrix (of DxL), w = bias vector (of Dx1).

% Any non-mandatory argument can be given the value [] to force it to take
% its default value.

% Copyright (c) 2015 by Ramin Raziperchikolaei and Miguel A. Carreira-Perpinan

function [h,Z,f] = ba(X,L,mu,Z,V,enum)

% Normalize data points to [0,1]
max_dims = max(X,[],1); min_dims = min(X,[],1); 
range_dims = max(max(max_dims-min_dims+eps));
X = bsxfun(@minus,X,min_dims); 
X = bsxfun(@rdivide,X,range_dims);

N = size(X,1);

% ---------- Argument defaults ----------
if ~exist('Z','var') || isempty(Z) [~,Z] = itq(X,L); end;
if ~exist('V','var') || isempty(V) V = X(randperm(N,200),:); end;
if ~exist('enum','var') || isempty(enum) enum = 16; end;
% ---------- End of "argument defaults" ----------

mkdir BAtemp		% Temporary directory for LIBLINEAR files

% Ground truth for the validation set
k = floor(0.002*N); dist = sqdist(V,X); [~,gt] = sort(dist,2);
gt = gt(:,1:k); oldP = 0;

% Initialization of binary codes, hash function and decoder
L = size(Z,2); 
Z = logical(Z); 
oldZ = false((size(Z)));
rZ = []; 
h = []; 
f = [];
SX = sparse(X);		% LIBLINEAR requires a sparse data matrix
i = 0;
while i < length(mu)
  i = i+1;

  oldf = f; 
  f = linftrain(double(Z),X);		% Train the decoder  

  % Determine which bits have not changed from the previous iteration
  do_h = sum(xor(Z,oldZ)); 
  do_h = do_h>0; 
  oldh = h;
  [h hX] = optenc(double(Z),SX,h,i~=1,do_h);	% Train the encoder (hash fcn)
  
  % Check precision on validation set before Z step
  newP = KNNPrecision(hX,linh(V,h),k,gt);
  if newP < oldP	  % Skip the step if the precision does not increase
    h = oldh; 
    f = oldf; 
    Z = oldZ; 
    rZ = oldrZ;
    idx = find(mu==mu(i)); 
    i = idx(end)+1;
    if i>length(mu) 
        disp('Stop because i > length(mu)');
        break; 
    end
  else
    oldP = newP;
  end
  
  % Train the binary codes (Z step)
  oldZ = Z; 
  oldrZ = rZ;
  if L < enum
    Z = Zenum(X,f,hX,mu(i));		% Enumeration
  else					% Alternating optimization
    % Initialization: truncated relaxed problem
    [Z rZ] = Zrelaxed(X,f,hX,mu(i),rZ);
    % Alternating optimization
    Z = Zaltopt(X,f,hX,mu(i),Z);
  end
  
  % Stop when the output of the hash function equals the binary codes
  if all(hX(:)==Z(:)) 
      disp('Stop because Z = h');
      break; 
  end
end
if (i == length(mu))
     disp('Stop because out of mu');
end
Z = hX;

% Unnormalise back hash function h
h.W = bsxfun(@rdivide,h.W,range_dims); h.w = h.w - h.W*min_dims';

delete('BAtemp/*'); rmdir BAtemp	% Remove the temporary directory

end

