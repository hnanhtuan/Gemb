% clc
% clear
rng('shuffle')

addpath('ITQ/', 'PCAH/', 'SH/', 'SpH/', 'utils/');
addpath(genpath('BA/'));
addpath(genpath('KMH/'));

%% Set parameters
% array of all hash code length. 
% (The program will fit gmm for all hash code length in parallel to speed up.)
num_bits = [16, 32, 64];        

% Dataset (cifar10, mnist, labelme)
dataset = 'mnist';

% Select a descriptor type ('vggfc7', 'gist')
% - 'vggfc7': descriptor is extracted from the activation of fully
% connected layer 7 of VGG net.
% - 'gist': the descriptor is GIST 512-D.
feature_type = 'gist';        
if strcmp(feature_type, 'vggfc7')
    variance_thresh = 0.65;
    pw = 0.05;
else
    variance_thresh = 0.85;
    pw = 0.15;
end
dataset = [dataset,'-', feature_type];

% Select hashing method ('itq', 'ba', 'kmh', 'sh', 'sph')
hash_method = 'itq';

fprintf(2,'================ GEMB + %s ======================\n', upper(hash_method));
%% LOAD DATASET
% The train(database) set (features/labels) and query set (features/labels) are 
% stored in 4 variables: train_features, train_labels, query_features, query_labels
% + train_features: n x d (num train samples x feature dim)
% + train_labels: n x 1 (num train samples x 1)
% + query_features: m x d (num query samples x feature dim)
% + query_labels: m x 1 (num query samples x 1)
fprintf('+ DATASET: %s\n', upper(dataset));
[train_features, train_labels, query_features, query_labels] = load_dataset( dataset );

%% RUN EMBEDDING
% zero-mean
avg = mean(train_features, 2);
X = bsxfun(@minus, train_features, avg);
Q = bsxfun(@minus, query_features, avg);

% Preprocess data to remove correlated features.
Xcov = X*X';
Xcov = (Xcov + Xcov')/(2*size(X, 2));
[U,S,~] = svd(Xcov);
k = size(S, 1); 
demo = sum(S(:));
temp = demo;
for i=k - 1:-1:1
    temp = temp - S(i, i);
    perVarRetain = temp/demo;
    if (perVarRetain < variance_thresh)
        k = i + 1;
        break
    end
end

disp(['+ REMAIN COMPONENTS: ', num2str(k)]);
X_pca = U(:,1:k)' * X;
Q_pca = U(:,1:k)' * Q;

tic
gmms = cell(length(num_bits), 1);
parfor idx=1:length(num_bits)
    gmms{idx} = fitgmdist(X_pca', num_bits(idx), 'CovarianceType', 'full', ...
                         'Options', statset('Display', 'off', 'MaxIter', 10, 'TolFun', 1e-6),...
                         'Start', 'plus', 'RegularizationValue', 0.001);
end
fprintf('\t Fit GMM in %.3fs\n', toc);

%% EVALUATION
disp('+ EVALUATION');
for idx=1:length(num_bits)
    gmm = gmms{idx};
    num_bit = num_bits(idx);
    X_pca_gemb = posterior(gmm, X_pca');
    Q_pca_gemb = posterior(gmm, Q_pca');
    
    % Power Normalization
    X_pca_gemb = sign(X_pca_gemb).*abs(X_pca_gemb).^pw;
    Q_pca_gemb = sign(Q_pca_gemb).*abs(Q_pca_gemb).^pw;

    % zero-mean
    X_pca_gemb_norm_avg = mean(X_pca_gemb);
    X_pca_gemb = bsxfun(@minus, X_pca_gemb, X_pca_gemb_norm_avg);
    Q_pca_gemb = bsxfun(@minus, Q_pca_gemb, X_pca_gemb_norm_avg);
    clear X_pca_gemb_norm_avg

    switch hash_method
        case 'itq'
            clear ITQparam
            ITQparam.nbits = num_bit;
            ITQparam = trainITQ(X_pca_gemb, ITQparam);   
            [~, ~, gallery_code] = compressITQ(X_pca_gemb, ITQparam);
            [~, ~, test_code]    = compressITQ(Q_pca_gemb, ITQparam);
        case 'ba'
            mu = 10^-5*2.^[1:1];
            [BA_h, gallery_code] = ba(X_pca_gemb, num_bit, mu);
                   test_code     = linh(Q_pca_gemb, BA_h);
        case 'kmh'
            b = 8; lambda = 10;
            kmh_num_iter = 50;    % num of iterations, 50 to 200. Usually 50 is enough.
            M = num_bit / b;      % num of subspaces
            
            [centers_table, R, sample_mean] = trainKMH(X_pca_gemb, M, b, kmh_num_iter, lambda);
            gallery_code = encode_KMH_2(X_pca_gemb, R, sample_mean, M, b, centers_table);
            test_code    = encode_KMH_2(Q_pca_gemb, R, sample_mean, M, b, centers_table);
        case 'sh'
            clear SHparam
            SHparam.nbits = num_bit;
            SHparam = trainPCAH(X_pca_gemb, SHparam);
            SHparam = trainSH(X_pca_gemb, SHparam);
            [~, ~, gallery_code] = compressSH(X_pca_gemb, SHparam);
            [~, ~, test_code]    = compressSH(Q_pca_gemb, SHparam);
        case 'sph'
            clear SpHparam
            SpHparam.nbits = num_bit;
            SpHparam.ntrain = size(X_pca_gemb, 1);
            SpHparam = trainSpH(X_pca_gemb, SpHparam);
            [gallery_code, test_code] = compressSpH([X_pca_gemb; Q_pca_gemb], SpHparam);
        otherwise 
            error('Unknown hashing method!');
    end
    
    [Pr2, Ptop1000, mAP] = evaluation(gallery_code, train_labels, test_code, query_labels);
         
    fprintf('  * %d bits: r=2: %f - top1k: %f - mAP: %f \n', num_bit, Pr2, Ptop1000, mAP);
end
