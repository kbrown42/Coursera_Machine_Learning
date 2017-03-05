function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and
%   sigma. You should complete this function to return the optimal C and
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;
C_vals = [.01, .03, .1, .3, 1, 3, 10, 30];
sigma_vals = C_vals;


% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example,
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using
%        mean(double(predictions ~= yval))
%
len = length(C_vals);
opts = zeros(len ^ 2,2);

cv_errs = zeros(size(opts, 1), 1);

for i = 1:length(C_vals)
  for j = 1:length(sigma_vals)
    opts((i - 1) * len + j, :) = [C_vals(i), sigma_vals(j)];
  end
end

for i = 1:size(opts, 1)
  fprintf('Training example %d', i)
  opt = opts(i, :);
  model = svmTrain(X, y, opt(1),  @(x1, x2) gaussianKernel(x1, x2, opt(2)));
  preds = svmPredict(model, Xval);
  cv_errs(i) = mean(double(preds ~= yval));

end

[m, ix] = min(cv_errs, [], 1);
m
ix
C = opts(ix, 1)
sigma = opts(ix, 2)
% =========================================================================

end
