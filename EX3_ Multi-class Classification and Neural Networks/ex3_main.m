%% Machine Learning Stanford- Exercise 3 : Multi-class Classification


%% Initialization
clear ; close all; clc

%% Load Data
load('ex3data1.mat')
% The matrices X and y will now be in MATLAB env

%% ==================== Part 1.2: Visualizing the data ====================
m = size(X, 1);
% Randomly select 100 data points to display.
rand_indices = randperm(m);
selectedPoints = X(rand_indices(1:100), :);
displayData(selectedPoints);

%% ==================== Part 1.3: Vectorizing logistic regression ====================
theta_t = [-2 ; -1; 1; 2];
X_t = [ones(5,1) reshape(1:15, 5, 3) / 10];
y_t = ([1;0;1;0;1] >= 0.5);
lamda_t = 3;
[J, grad] = lrCostFunction(theta_t, X_t, y_t, lamda_t);

fprintf('Cost: %f | Expected cost: 2.534819\n', J);
fprintf('Gradients:\n'); fprintf('%f\n', grad);
fprintf('Expected gradients:\n 0.146561\n -0.548558\n 0.724722\n 1.398003');
%% ==================== Part 1.4: One-vs-all classication ====================
num_labels = 10;    % 10 labels, from 1 to 10
lambda = 0.1;
[all_theta] = oneVsAll(X, y, num_labels, lambda);

%% ==================== Part 1.4.1: One-vs-all prediction ====================
[pred, ~] = predictOneVsAll(all_theta, X);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

%% ==================== Part 2: Neural Networks ====================

% Load savd matrices from file
load('ex3weights.mat')
% Theta1 has sie 25 x 401
% Theta2 has size 10 x 26
pred = predict(Theta1, Theta2, X);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

%  Randomly permute examples
rp = randi(m);
% Predict
pred = predict(Theta1, Theta2, X(rp,:));
fprintf('\nNeural Network Prediction: %d (digit %d)\n', pred, mod(pred, 10));
% Display 
displayData(X(rp, :));   