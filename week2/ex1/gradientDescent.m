function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
alpha
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    h = X * theta;
    errs = (h - y);

    d_cost = errs' * X;  % Gives a 1 x 2 matrix: each x_j times alpha
    grad = (alpha * ((1/m) * d_cost')');
    theta = theta - grad';    





    % ============================================================

    % Save the cost J in every iteration    
    c = computeCost(X, y, theta);
    J_history(iter) = c;

end

end
