%% Machine Learning Stanford- Exercise 6 : Support Vector Machines

%% Initialization
clear ; close all; clc

%% Load Data
load ('ex6data1.mat');
% The matrices X and y will now be in MATLAB env
%% ==================== Part 1.1: Visualizing the data ====================
m = size(X, 1);
% Plot training data
plotData(X,y);

%% SVM Training
C = 1;
model = svmTrain(X, y, C, @linearKernel, 1e-3, 20);
visualizeBoundaryLinear(X, y, model);

%% 1.2.1: SVM with Gaussian Kernels (Dataset 1)

x1 = [1 2 1]; x2 = [0 4 -1]; 
sigma = 2;
sim = gaussianKernel(x1, x2, sigma);
fprintf('Gaussian Kernel between x1 = [1; 2; 1], x2 = [0; 4; -1], sigma = %f : \n\t%g\n', sigma, sim);


%% 1.2.2: Example dataset 2
% Load dataset 2
load('ex6data2.mat');

% Plot training data
plotData(X, y);

%% SVM with Gaussian Kernels (Dataset 2)
% SVM Parameters
C = 1; sigma = 0.1;

% We set the tolerance and max_passes lower here so that the code will run faster. However, in practice, 
% you will want to run the training to convergence.
model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
visualizeBoundary(X, y, model);

%% 1.2.3: Example dataset 3
% Load from ex6data3
load('ex6data3.mat');

plotData(X, y);

%% SVM with Gaussian Kernels (Dataset 2)
% Try different SVM Parameters here
[C, sigma] = dataset3Params(X, y, Xval, yval);

% Train the SVM
model = svmTrain(X, y, C, @(x1, x2)gaussianKernel(x1, x2, sigma));
visualizeBoundary(X, y, model);

%% 2: Spam Classification

%% Initialization
clear;

%% 2.1.1: Vocabulary list

% Extract Features
file_contents = readFile('emailSample1.txt');
word_indices  = processEmail(file_contents);
% Print Stats
disp(word_indices)

%% 2.2: Extracting features from emails
% Extract Features
features = emailFeatures(word_indices);

% Print Stats
fprintf('Length of feature vector: %d\n', length(features));
fprintf('Number of non-zero entries: %d\n', sum(features > 0));

%% 2.3: Training SVM for spam classification

% Load the Spam Email dataset
% You will have X, y in your environment
load('spamTrain.mat');
C = 0.1;
model = svmTrain(X, y, C, @linearKernel);

p = svmPredict(model, X);
fprintf('Training Accuracy: %f\n', mean(double(p == y)) * 100);

% Load the test dataset
% You will have Xtest, ytest in your environment
load('spamTest.mat');

p = svmPredict(model, Xtest);
fprintf('Test Accuracy: %f\n', mean(double(p == ytest)) * 100);

%% 2.4: Top predictors for spam
% Sort the weights and obtin the vocabulary list
[weight, idx] = sort(model.w, 'descend');
vocabList = getVocabList();
for i = 1:15
    if i == 1
        fprintf('Top predictors of spam: \n');
    end
    fprintf('%-15s (%f) \n', vocabList{idx(i)}, weight(i));
end