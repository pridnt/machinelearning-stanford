function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% calculate hypothesis
h = X * theta;

% calculate cost
% calculate cost without regular term
J = sum((h - y) .^2) / (2 * m);

% calculate gradient
% calculate gradient without regular term
grad = (X' * (h - y)) / m;

% Setting theta(1) equals to 0
theta(1) = 0;

% calculate regular term cost
J_regTerm = lambda / (2 * m) * sum(theta.^2);
% calculate total cost
J = J + J_regTerm;

% calculate regular term gradient
grad_regTerm = (lambda / m) * theta;
% calculate total gradient
grad = grad + grad_regTerm;

% =========================================================================

grad = grad(:);

end
