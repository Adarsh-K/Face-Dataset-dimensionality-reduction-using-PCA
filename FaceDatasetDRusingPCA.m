%% Face Dataset dimensionality reduction using PCA
%% Initialization
clear ; close all; clc
%% =============== Visualizing Face Data =============
%  I'll start by first visualizing the dataset.
%
fprintf('\nLoading face dataset.\n\n');

load ('ex7faces.mat')

displayData(X(1:100, :)); %  Display the first 100 faces in the dataset

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =========== PCA on Face Data: Eigenfaces  ===================
%  Now lets run PCA and visualize the eigenvectors which are in this case eigenfaces
%  Also lets display the first 36 eigenfaces.
%
fprintf(['\nRunning PCA on face dataset.\n' ...
         '(this might take a minute or two ...)\n\n']);

[X_norm, mu, sigma] = featureNormalize(X); % Feature normalising

[U, S] = pca(X_norm);

displayData(U(:, 1:36)');

fprintf('Program paused. Press enter to continue.\n');
pause;


%% ============= Dimension Reduction for Faces =================
%  Now I'll project images to the eigen space using the top k eigenvectors

fprintf('\nDimension reduction for face dataset.\n\n');

K = 100; % Could change this K
Z = projectData(X_norm, U, K);

fprintf('The projected data Z has a size of: ')
fprintf('%d ', size(Z));

fprintf('\n\nProgram paused. Press enter to continue.\n');
pause;

%% ==== Visualization of Faces after PCA Dimension Reduction ====
%  Lets compare to the original input, which is also displayed

fprintf('\nVisualizing the projected (reduced dimension) faces.\n\n');

K = 100;
X_rec  = recoverData(Z, U, K);

% Display normalized data
subplot(1, 2, 1);
displayData(X_norm(1:100,:));
title('Original faces');
axis square;

% Display reconstructed data from only k eigenfaces
subplot(1, 2, 2);
displayData(X_rec(1:100,:));
title('Recovered faces');
axis square;

fprintf('Program paused. Press enter to continue.\n');
pause;

for i = 1:100
    % Display 
    fprintf('\nDisplaying Example Image\n');
    
    % Display normalized data
    subplot(1, 2, 1);
    displayData(X_norm(i,:));
    title('Original faces');
    
    % Display reconstructed data from only k eigenfaces
    axis square;subplot(1, 2, 2);
    displayData(X_rec(i,:));
    title('Recovered faces');
    axis square;

    % Pause with quit option
    s = input('Paused - press enter to continue, q to exit:','s');
    if s == 'q'
      break
    end
end