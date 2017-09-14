function BincodeX = encode_KMH_2(X, R, sample_mean, M, b, centers_table)

X = project(X, R, sample_mean);

D = size(X,2);
d = D / M;

lut = generate_lut(b);

codeX = zeros(size(X,1), M*b);

for m = 1:M    
    distX=sqdist(X(:,(1:d)+(m-1)*d)',centers_table{m}');

    [distX, idx_centers]=min(distX,[],2);
    clear distX;
    
    codeX(:, (1:b)+(m-1)*b) = lut(idx_centers,:);
end
BincodeX = codeX;
% BincodeX = compactbit(codeX);
end