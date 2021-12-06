%% Machine Learning Stanford- Exercise 2:
% x1  refers to the Exam 1 Score
% x2  refers to the Exam 2 Score
% y refers to admission decision


%% Initialization
clear ; close all; clc

%% Load Data
data = load('ex2data1.txt');
X = data(:, [1, 2]);
y = data(:, 3);
%% ==================== Part 1.1: Visualizing the data ====================

% Plotting data
plotData(X,y);

hold on;

% Labels and Legend
xlabel('Exam 1 Score');
ylabel('Exam 2 Score');

% Specified in plot order
legend('Admitted', 'Not admitted');
hold off;

fprintf('\nProgram paused. Press enter to continue.\n');
pause;


%% ==================== Part 1.2.1: Warmup exercise: sigmoid function ====================
fprintf("Sigmoid\n");
sigmoid(0.5)

%% ==================== Part 1.2.2: Cost function and gradient ====================

% Setup the data matrix appropriately.
[m, n] = size(X);
% Add intercept term to X.
X = [ones(m, 1) X];

% Initialize the fitting parameters
initial_theta = zeros(n+1, 1);

% Compute and display the initial cost and gradient
[cost, grad] = costFunction(initial_theta, X, y);
fprintf('Cost at initial theta (zeros): %f\n', cost);
disp('Gradient at initial theta (zeros):'); 
disp(grad);

% Compute and display cost and gradient with non-zero theta
test_theta = [-24; 0.2; 0.2];
[cost, grad] = costFunction(test_theta, X, y);
fprintf('\nCost at non-zero test theta: %f\n', cost);
disp('Gradient at non-zero theta:'); disp(grad);

%% ==================== Part 1.2.3: Learning parameters using fminunc ====================

% Set options for fminunc
options = optimoptions(@fminunc, 'Algorithm', 'Quasi-Newton', 'GradObj', 'on', 'MaxIter', 400);

% Run fminunc to obtain the optimal theta
% This function will return theta and the cost.
[theta, cost] = fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);

% Print theta
fprintf('Cost at theta found by fminunc: %f\n', cost);
disp('theta:');disp(theta);

% Plot Boundary
plotDecisionBoundary(theta, X, y);
% Add some labels 
hold on;
% Labels and Legend
xlabel('Exam 1 score')
ylabel('Exam 2 score')
% Specified in plot order
legend('Admitted', 'Not admitted')
hold off;

%% ==================== Part 1.2.4: Evaluating logistic regression ====================
%  Predict probability for a student with score 45 on exam 1  and score 85 on exam 2 
prob = sigmoid([1 45 85] * theta);
fprintf('For a student with scores 45 and 85, we predict an admission probability of %f\n\n', prob);
% Compute accuracy on our training set
p = predict(theta, X);
fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);


%% ==================== Part 2.1: Visualizing the data ====================
%  The first two columns contains the X values and the third column
%  contains the label (y).
data = load('ex2data2.txt');
X = data(:, [1, 2]); y = data(:, 3);

plotData(X, y);
% Putting labels
hold on;
% Labels and Legend
xlabel('Microchip Test 1');
ylabel('Microchip Test 2');
% Specified in plot order
legend('y = 1', 'y = 0');
hold off;

%% ==================== Part 2.2: Feature Mapping ====================

% Add Polynomial Features
% Note that mapFeature also adds a column of ones for us, so the intercept term is handled
X = mapFeature(X(:,1), X(:,2));
disp(X);

%% ==================== Part 2.3: Cost Function and Gradient ====================
% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1);

% Set regularization parameter lambda to 1
lambda = 1;

% Compute and display initial cost and gradient for regularized logistic regression
[cost, grad] = costFunctionReg(initial_theta, X, y, lambda);
fprintf('Cost at initial theta (zeros): %f\n', cost);
fprintf('Expected cost (approx): 0.693\n');
fprintf('Gradient at initial theta (zeros) - first five values only:\n');
fprintf(' %f \n', grad(1:5));
fprintf('Expected gradients (approx) - first five values only:\n');
fprintf(' 0.0085\n 0.0188\n 0.0001\n 0.0503\n 0.0115\n');

% Compute and display cost and gradient with all-ones theta and lambda = 10
test_theta = ones(size(X,2),1);
[cost, grad] = costFunctionReg(test_theta, X, y, 10);
fprintf('\nCost at test theta (with lambda = 10): %f\n', cost);
fprintf('Expected cost (approx): 3.16\n');
fprintf('Gradient at test theta - first five values only:\n');
fprintf(' %f \n', grad(1:5));
fprintf('Expected gradients (approx) - first five values only:\n');
fprintf(' 0.3460\n 0.1614\n 0.1948\n 0.2269\n 0.0922\n');


legend('Admitted', 'Not admitted')
hold off;

%% Learning Parameteres using fminunc

% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1);

% Set regularization parameter lambda to 1 (you should vary this)
lambda = 0;

% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 400);

% Optimize
[theta, J, exit_flag] = ...
	fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);

% Plot Boundary
plotDecisionBoundary(theta, X, y);
hold on;
title(sprintf('lambda = %g', lambda))

% Labels and Legend
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')

legend('y = 1', 'y = 0', 'Decision boundary')
hold off;

% Compute accuracy on our training set
p = predict(theta, X);

fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);



