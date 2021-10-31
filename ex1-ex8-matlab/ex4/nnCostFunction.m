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
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



% -------------------------------------------------------------
% PART 1
% Add bias to the X data matrix
a1 = [ones(m, 1) X]';

% Calculate L2 output
z2 = Theta1*a1;
a2 = sigmoid(z2)';

%add bias term to the output of L2
a2 = [ones(size(a2,1), 1) a2]';

%perform ouput layer operations
z3 = Theta2*a2;
a3 = sigmoid(z3);

% Code ground truth values
y_coded = zeros(m, num_labels); % 5000 x10 matrix
%set 1 on the column that inidcates the numerical value
for i= 1:m
    y_coded(i, y(i))=1;
end

%===== COST ===============
cost= calculateCost(a3, y_coded, m);
J = regularizeCost(cost, Theta1, Theta2, lambda, m);

% PART 2: BACKPROPAGATION ===============================
for t = 1:m
    %perform forward propagation
    a1 = [1 X(t, :)]'
    z2 = Theta1* a1;
    a2 = sigmoid(z2);
    a2 = [1; a2];    
    z3 = Theta2*a2;
    a3 = sigmoid(z3);
    
    % cost for output layer (L3)
    delta_3 = a3 - y_coded(t, :)';
    
    %cost for hidden layer (L2). multiplies each theta column by the error of the next layer
    delta_2 =  (Theta2' * delta_3);
    z2 = [1; z2]; 
    delta_2 = delta_2.*sigmoidGradient(z2);
    %remove the gradient of bias unit.
    delta_2 = delta_2(2:end);
    
    % acumulate gradient (D)
    Theta2_grad = Theta2_grad + delta_3*a2';
    Theta1_grad = Theta1_grad + delta_2*a1';
    
end

Theta2_grad = Theta2_grad/m;
Theta1_grad = Theta1_grad/m;


% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end


% =====FUNCTIONS===================================

function J = calculateCost (y_hat, y_coded, m)
J=0;

cost= (-y_coded)' .* log(y_hat) - (1-y_coded)' .* log(1-y_hat); 
k_sum_error = sum(cost);
m_sum_error = sum(k_sum_error);
J = (1/m) * m_sum_error;
end

function J = regularizeCost(cost, Theta1, Theta2, lambda, m)
J = 0
t1 = sum(sum(Theta1(:, 2:end).^2));
t2 = sum(sum(Theta2(:, 2:end).^2));
regularization_term = (lambda/(2*m)*(t1+t2));
J = cost + regularization_term;
end
