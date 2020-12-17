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

% Setup some useful variables
m = size(X, 1);

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

X1 = [ones(m, 1) X];
A2 = sigmoid(Theta1 * X1');
A2 = [ones(size(A2,2), 1) A2'];
A3 = sigmoid(Theta2 * A2');
[x, p] = max(A3', [], 2);

sum2 = 0;
for i=1:m
  sum1 = 0;
  for k=1:num_labels
    % vector
    valY = y(i) == k;
    valPred = A3(k,i);
    value = (-valY * log(valPred)) - ((1 - valY) * (log(1 - valPred)));
    sum1 += value;
  endfor

  sum2 += sum1;
endfor

t1s = Theta1(:,2:end);
t2s = Theta2(:,2:end);
reg = (lambda/(2*m)) * (sum(sum(t1s.^2)) +  sum(sum(t2s.^2)));

J = (1/m * sum2) + reg;

%
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

X2 = [ones(m, 1) X];
for t=1:m

  a1 = X2'(:,t);
  a2 = [1; sigmoid(Theta1 * a1)];
  a3 = sigmoid(Theta2 * a2);

  yt = 1:size(a3,1);
  yt = yt == y(t);

  d3 = a3 - yt';
  d2 = (Theta2' * d3) .* [1; sigmoidGradient(Theta1 * a1)];
  d2 = d2(2:end);

  Theta1_grad = Theta1_grad + (d2 * a1');
  Theta2_grad = Theta2_grad + (d3 * a2');
endfor

Theta1_grad = ((1/m) * Theta1_grad);
Theta2_grad = ((1/m) * Theta2_grad);

reg1 = ((lambda/m) * Theta1);
reg2 = ((lambda/m) * Theta2);

Theta1_grad(:, 2:end) += reg1(:, 2:end);
Theta2_grad(:, 2:end) += reg2(:, 2:end);

%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
