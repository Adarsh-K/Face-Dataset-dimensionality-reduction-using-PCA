function [U, S] = pca(X)
%   [U, S, X] = pca(X) computes eigenvectors of the covariance matrix of X
%   Returns the eigenvectors U, the eigenvalues (on diagonal) in S
%

[m, n] = size(X);

U = zeros(n);
S = zeros(n);

con_mat=(1/m)*X'*X;
[U,S,V]=svd(con_mat);

end
