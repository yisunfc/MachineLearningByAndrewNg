function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
n=length(grad)
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

for i = 1:m
	temp = sigmoid((X(i, :) * theta)');
	J = J - y(i) * log(temp) - (1 - y(i)) * log(1 - temp);
end
for j = 2:n
	J = J + lambda / 2 * theta(j)^2
end

J = J / m

grad = 1 / m * ((sigmoid(theta' * X') - y') * X);
for j = 2:n
	grad(j) = grad(j) + lambda / m * theta(j)
end


% =============================================================

end
