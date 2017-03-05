function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% fprintf('Thetas: \n')
% size(Theta1)
% size(Theta2)
% Setup some useful variables
m = size(X, 1);
X = [ones(m, 1) X];

% fprintf('size of input\n')
% size(X)
% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

% 5000 x 401 * 401 x 26
layer1_in = X * Theta1';
a1 = [ones(m, 1) sigmoid(layer1_in)];  % 5000 x 26

% 5000 x 26 * 26 x 10 -> 5000 x 10
layer2_in = a1 * Theta2';
a2 = sigmoid(layer2_in); % output values
% predictions
% [val, p] = max(a2, [], 2);

% compute cost function
%  y = 5000 x 1 -> 5000 x 10
% a2 = 5000 x 10
targets = zeros(m, num_labels);
ind = sub2ind(size(targets), 1:m, y');
targets(ind) = 1;


J  = (1 / m) * sum(...
                 sum( ...
                 (((-targets) .* log(a2)) - ((1 - targets) .* log(1 - a2)))...
                 )...
               );

tTheta1 = Theta1(:, 2:end);
tTheta2 = Theta2(:, 2:end);

tTheta1 = reshape(tTheta1, size(tTheta1, 1) * size(tTheta1, 2), 1);
tTheta2 = reshape(tTheta2, size(tTheta2, 1) * size(tTheta2, 2), 1);

thetas = [tTheta1; tTheta2];
bias = (lambda / ( 2 * m)) * sum(thetas .^ 2);

J = J + bias;


% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.
%

a_1 = X;

z_2 = X * Theta1';
a_2 = sigmoid(z_2);
a_2 = [ones(m, 1) a_2];

z_3 = a_2 * Theta2';
a_3 = sigmoid(z_3);

d_3 = a_3 - targets;

d_2 = (d_3 * Theta2)(:, 2:end) .* (sigmoidGradient(z_2));

delta_2 = d_3' * a_2;

delta_1 = d_2' * a_1;

Theta1_grad = (1/m) * delta_1;
Theta2_grad = (1/ m ) * delta_2;


% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

bias1 = (lambda / m) * Theta1(:, 2:end);
bias2 = (lambda / m) * Theta2(:, 2:end);

Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + bias1;
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + bias2;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

% fprintf('size of my gradient vector: \n')
% size(grad)
end
