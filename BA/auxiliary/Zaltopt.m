% Z = Zaltopt(X,f,V,mu,[Z,g,maxit])
%
% Binary autoencoder Z step: alternating optimization.
%
% Optimizes over the binary codes Z by alternating optimization:
%   min_Z{ |X - f(Z)|² + µ.|Z - V|² } s.t. Z in {0,1}
%
% In:
%   X: NxD matrix.
%   f: mapping from z to x.
%   V: NxL binary matrix.
%   mu: positive scalar.
%   Z: NxL binary matrix (initial Z). Default: V.
%   g: size of blocks in alternating optimization. Default: 1.
%   maxit: maximal number of iterations. Default: 100.
% Out:
%   Z: NxL binary matrix.
%
% Any non-mandatory argument can be given the value [] to force it to take
% its default value.

% Copyright (c) 2015 by Ramin Raziperchikolaei and Miguel A. Carreira-Perpinan

function Z = Zaltopt(X,f,V,mu,Z,g,maxit)

% ---------- Argument defaults ----------
if ~exist('Z','var') || isempty(Z) Z = V; end;
if ~exist('g','var') || isempty(g) g = 1; end;
if ~exist('maxit','var') || isempty(maxit) maxit = 100; end;
% ---------- End of "argument defaults" ----------

[N,L] = size(Z);

% Zaltopt is usually called with g=1, so we implement this case efficiently
if g==1
  W = f.W; w = f.w; [Q R] = qr(W,0); X = bsxfun(@minus,X,w')*Q; 
  % numFixed keeps, for each data point, the number of consecutive bits that
  % have not changed in the alternating optimization.
  numFixed = zeros(1,N);
  parfor n = 1:N
    z = Z(n,:); v = V(n,:); x = X(n,:);
    i=1; counter=1;
    % Loop over bit positions 1 to L repeatedly until no change happens
    % for L consecutive bits.
    while numFixed(n) ~= L && counter < (maxit*L)
      i = mod(i-1,L)+1;		% generate index i=1 to L
      e1 = (norm(x-z*R').^2) + (mu)*sum(xor(z,v));
      z(i) = ~z(i); e2 = (norm(x-z*R').^2) + (mu)*sum(xor(z,v));
      if e1 < e2		% no change in bit position i
        z(i) = ~z(i);
        numFixed(n) = numFixed(n) + 1;
      else			% change in bit position i
        numFixed(n) = 0;
      end      
      i=i+1; counter=counter+1;
    end
    Z(n,:) = z;
  end
else				  % alternating optimization for g>1
  numBlocks = L/g; B = binset(g);
  numFixed = zeros(1,N); W = f.W; w = f.w; [Q R] = qr(W,0);
  X = bsxfun(@minus,X,w')*Q; V = logical(V);
  parfor n = 1:N
    z = Z(n,:); x = X(n,:); h = V(n,:);
    i=1;
    while numFixed(n) ~= numBlocks
      i = mod(i-1,numBlocks)+1;
      sIndex = (i-1)*g + 1; eIndex = (i)*g;
      zRep = repmat(z,2^g,1); zRep(:,sIndex:eIndex) = B; RZ = zRep*R';
      obj = sum(bsxfun(@minus,x,RZ).^2,2) + mu*sum(bsxfun(@xor,zRep,h),2);
      [~,idx] = min(obj);
      if (~isequal(zRep(idx,:),z))
        z = zRep(idx,:); numFixed(n) = 0;
      else
        numFixed(n) = numFixed(n) + 1;
      end
      i=i+1;
    end
    Z(n,:) = z;
  end
end

end

