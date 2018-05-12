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
min_pred_error = 10;
C_s = [0.01, 0.03, 0.1 , 0.3, 1, 3, 10, 30];
Sigmas = [0.01, 0.03, 0.1 , 0.3, 1, 3, 10, 30];
for CC = 1:length(C_s)
      for sig = 1:length(Sigmas)
            model= svmTrain(X, y, C_s(CC), @(x1, x2) gaussianKernel(x1, x2, Sigmas(sig)));
            predictions = svmPredict(model, Xval);
            pred_error = mean(double(predictions ~= yval));
            fprintf(['Prediction error for C = %f and sigma = %f : %f  .\n'], C_s(CC), Sigmas(sig), pred_error);
            if pred_error<min_pred_error
                  fprintf(['min_pred_error = %f  .\n'], min_pred_error);
                  C=C_s(CC);
                  sigma = Sigmas(sig);
                  fprintf(['NEW C = %f , sigma = %f .\n'], C, sigma);
                  min_pred_error = pred_error
            endif
      endfor
endfor              
               





% =========================================================================

end
