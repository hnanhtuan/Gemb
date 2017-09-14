% Z = Zenum(X,f,V,mu)
%
% Binary autoencoder Z step: enumeration.
%
% Optimizes over the binary codes Z by enumeration:
%   min_Z{ |X - f(Z)|² + µ.|Z - V|² } s.t. Z in {0,1}
%
% In:
%   X: NxD matrix.
%   f: mapping from z to x.
%   V: NxL binary matrix.
%   mu: positive scalar.
% Out:
%   Z: NxL binary matrix.

% Copyright (c) 2015 by Ramin Raziperchikolaei and Miguel A. Carreira-Perpinan

function Z = Zenum(X,f,V,mu)
    
[N L] = size(V); Z = V; B = binset(L);
W = f.W; w = f.w; [Q R] = qr(W,0); X = bsxfun(@minus,X,w')*Q; RZ = B*R';

% N independent problems, optimize each one separately and in parallel
parfor n=1:N
  x = X(n,:); v = V(n,:);
  % nth problem: min_z{ |x - f(z)|² + µ.|z - v|² }
  idx1 = bin2dec(int2str(v))+1;
  bound = sum((x-RZ(idx1,:)).^2);
  err1 = mu*sum(bsxfun(@xor,B,v),2);
  idx2 = find(err1<=bound);
  if numel(idx2) == 1 continue; end
  [~,idx3] = min(sum(bsxfun(@minus,x,RZ(idx2,:)).^2,2) + err1(idx2));
  Z(n,:) = B(idx2(idx3),:);
end

end

