% [h,hX] = optenc(Z,X,[h,warm,do_h])
%
% Train the encoder (hash function) given input data and binary codes.
%
% The hash function h consists of L binary linear SVMs (one per code bit).
% We train each SVM using LIBLINEAR.
%
% Notes:
% - Warm-start means that, when training the hash function h, we initialize
%   the training to the hash function resulting from previous MAC iteration.
%   This way, training is faster than if using an arbitrary initial h.
% - When using warm-start, LIBLINEAR uses text files to pass information: it
%   puts all the information about the trained SVMs in text files (with names
%   tmph001, tmph002... for bits 1, 2...) and initializes itself by reading
%   from them. In our code, the input argument h contains the same information
%   as the text files but in matrix format so we can use it in our Matlab code.
%
% In:
%   Z: NxL binary matrix containing N L-dim binary data points rowwise.
%   X: NxD *sparse* matrix containing N D-dim data points rowwise.
%   h: (struct) hash function (L binary SVMs) trained in the previous
%      iteration of the MAC algorithm.
%   warm: 1 if we use warm-start to initialize h, 0 otherwise. Default: 0.
%   do_h: 1xL binary vector, do_h(l) = 1 if we need to train the lth hash
%      function. Default: ones.
% Out:
%   h: (struct) hash function (L binary SVMs).
%   hX: NxL binary matrix containing the output of the hash function for
%      each training point.

% Copyright (c) 2015 by Ramin Raziperchikolaei and Miguel A. Carreira-Perpinan

function [h,hX] = optenc(Z,X,h,warm,do_h)

L = size(Z,2); D = size(X,2);

% ---------- Argument defaults ----------
if ~exist('h','var') || isempty(h) h.W = zeros(L,D); h.w = zeros(L,1); end;
if ~exist('warm','var') || isempty(warm) warm = 0; end;
if ~exist('do_h','var') || isempty(do_h) do_h = ones(1,L); end;
% ---------- End of "argument defaults" ----------

W = h.W; w = h.w;	% unpack struct so we can assign variables in parfor
for j=find(do_h)
  % Call LIBLINEAR's "train" with appropriate parameters:
  LIBLINEARopt = [' tmph' num2str(j,'%03d')];
  if warm LIBLINEARopt = [' -i' ' tmph' num2str(j,'%03d') LIBLINEARopt]; end
  model = train(Z(:,j),X,['-e .001 -s 2 -B 1 -q -c 100' LIBLINEARopt]);
  % Extract SVM parameters from LIBLINEAR output:
  tempw = (model.Label(1)*2 - 1)*model.w;
  W(j,:) = tempw(1:end-1)'; w(j,1) = tempw(end);
end
h.type = 'linh'; h.w = w; h.W = W;

if nargout > 1 hX = linh(X,h); end

end

