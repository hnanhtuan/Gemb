% [h,Z,f] = bfa(X,L,[Z,V,enum,maxit]) Binary Factor Analysis (BFA)
%
% Train a binary factor analysis using a MAC algorithm. The encoder can be
% used as a binary hash function for information retrieval.
%
% See usage instructions in ba.m.
%
% In:
%   X,L,Z,V,enum: as in ba.m. Default for Z: tPCA.
%   maxit: maximal number of iterations. Default: 20.
% Out:
%   h,Z,f: as in ba.m.

% Any non-mandatory argument can be given the value [] to force it to take
% its default value.

% Copyright (c) 2015 by Ramin Raziperchikolaei and Miguel A. Carreira-Perpinan

function [h,Z,f] = bfa(X,L,Z,V,enum)

% Normalize data points to [0,1]
max_dims = max(X,[],1); min_dims = min(X,[],1); 
range_dims = max(max(max_dims-min_dims+eps));
X = bsxfun(@minus,X,min_dims); X = bsxfun(@rdivide,X,range_dims);

N = size(X,1);

% ---------- Argument defaults ----------
if ~exist('Z','var') || isempty(Z) [~,Z] = itq(X,L,0); end;
if ~exist('V','var') || isempty(V) V = X(randperm(N,200),:); end;
if ~exist('enum','var') || isempty(enum) enum = 10; end;
if ~exist('maxit','var') || isempty(maxit) maxit = 20; end;
% ---------- End of "argument defaults" ----------

mkdir BAtemp		% Temporary directory for LIBLINEAR files

% Ground truth for the validation set
k = floor(0.01*N); dist = sqdist(V,X); [~,gt] = sort(dist,2);
gt = gt(:,1:k); oldP = 0;

% Initialization of binary codes, hash function and decoder
L = size(Z,2); Z = logical(Z); oldZ = false((size(Z)));
rZ = []; h = [];
SX = sparse(X);		% LIBLINEAR requires a sparse data matrix
i = 0;
while i < maxit
  i = i+1;

  f = linftrain(double(Z),X);			% Train the decoder

  % Determine which bits have not changed from the previous iteration
  do_h = sum(xor(Z,oldZ)); do_h = do_h>0;
  [h hX] = optenc(double(Z),SX,h,i~=1,do_h);	% Train the encoder (hash fcn)
  
  % Check precision on validation set before Z step
  newP = KNNPrecision(hX,linh(V,h),k,gt);
  if newP < oldP	  % Stop when new precision worse than previous one
    break
  else
    oldP = newP;
  end
  
  % Train the binary codes (Z step)
  oldZ = Z;
  if L <= enum
    Z = Zenum(X,f,hX,0);		% Enumeration
  else					% Alternating optimization
    % Initialization: truncated relaxed problem
    [Z rZ] = Zrelaxed(X,f,hX,0,rZ);
    % Alternating optimization
    Z = Zaltopt(X,f,hX,0,Z);
  end
  
  % Stop when the output of the hash function equals the binary codes
  % or when there is no change in binary codes
  if (all(hX(:)==Z(:))) || (all(oldZ(:)==Z(:))) break; end
end
Z = hX;

% Unnormalise back hash function h
h.W = bsxfun(@rdivide,h.W,range_dims); h.w = h.w - h.W*min_dims';

delete('BAtemp/*'); rmdir BAtemp	% Remove the temporary directory

end

