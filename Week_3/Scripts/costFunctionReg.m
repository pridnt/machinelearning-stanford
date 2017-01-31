function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% regular theta
theta_reg = theta(2:size(theta, 1));

% calculate hypothesis
h = sigmoid(X*theta);

% calculate cost
% calculate cost without regular term
J = sum((-y' * log(h)) - ((1 - y)' * log(1 - h))) / m;
% calculate regular term cost
J_regTerm = lambda / (2 * m) * sum(theta_reg.^2);
% calculate total cost
J = J + J_regTerm;


% calculate gradient
% calculate gradient without regular term
grad = (X' * (h - y)) / m;
% calculate regular term gradient
grad_regTerm = (lambda / m) * [0; theta_reg];
% calculate total gradient
grad = grad + grad_regTerm;

% =============================================================

end
