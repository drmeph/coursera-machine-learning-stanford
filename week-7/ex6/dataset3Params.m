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
sigma = 0.01;

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

if false

  values = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
  %values = [0.01; 0.03; 0.1];


  x1 = [1 2 1]; x2 = [0 4 -1];
  results = zeros(64,3);
  index = 1;

  for i=1:size(values,1)
    Cvalue = values(i);

    for j=1:size(values,1)
      SigValue = values(j);
      model= svmTrain(X, y, Cvalue, @(x1, x2) gaussianKernel(x1, x2, SigValue));
      predictions = svmPredict(model, Xval);
      error = mean(double(predictions ~= yval));

      results(index, :) = [Cvalue SigValue error];
      index++;
    endfor
  endfor

  results
  [val, index] = min(results(:,3))
  results(index, :)

endif

C = 1;
sigma = 0.1;





% =========================================================================

end
