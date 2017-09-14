function PCAHparam = trainPCAH(X, PCAHparam)

% Input:
%          X: features matrix [Nsamples, Nfeatures]
%          PCAHparam.nbits: number of bits (nbits do not need to be a multiple of 8)
%
% Output:
%             PCAHparam:
%                 PCAHparam.pcaW---principle component projection
%                 PCAparam.nbits---encoding length

npca = PCAHparam.nbits;
[pc, ~] = eigs(cov(X), npca);   % return the k largest magnitude eigenvalues
PCAHparam.pcaW = pc; % no need to remove the mean

% fprintf('PCAH training process has finished\r');