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

% cost function: X: 5000X400, Theta1 :  25 X 401, Theta2 : 10X26

X = [ones(m, 1) X];

for i = 1: m

  a1 = X(i,:)'; % ith example , it is 401X1
  
  z2 = Theta1 * a1; % 25X1 

  a2 = sigmoid(z2); % 25X1

  % expend a2 with bias unit
  a2 = [ones(1, columns(a2)); a2];  % 26 X 1 , add 1 to row 1
  
  z3 = Theta2 * a2 ; % 10 X1 

  a3 =  sigmoid(z3); % 10 X 1
  
  delta3 = zeros(num_labels,1); % 10 X 1 

  for k = 1 : num_labels
    
    y_k = y(i)==k;
    
    J = J - (y_k*log(a3(k)) + (1-y_k)*log(1-a3(k)));
        
    %populate error 3
    delta3(k) = a3(k)- y_k;
    
  endfor
  
  %back propagation
  t2 = Theta2' * delta3; % 26 X 1
  
  delta2 = t2(2:end).* sigmoidGradient(z2); % 25 X 1
  
  Theta1_grad = Theta1_grad + delta2 * (a1)'; % 25 X 401  
     
  Theta2_grad = Theta2_grad + delta3 * (a2)'; % 10X26
  

endfor

Theta1_grad = 1/m * Theta1_grad;
Theta2_grad = 1/m * Theta2_grad;

% add regularized term

Theta1_grad  = Theta1_grad +lambda/m *Theta1;
Theta2_grad  = Theta2_grad +lambda/m *Theta2;


J = J/m;

% regularization of cost function
% for layer 1, Theta1 25*401, except first column 
r_1 =0;
for i = 1: rows(Theta1)
  for j = 2 : columns(Theta1)
    r_1 = r_1 + Theta1(i,j) *Theta1(i,j);
  endfor
  % take out bias unit in grad
  Theta1_grad(i,1) =  Theta1_grad(i,1)- lambda/m * Theta1(i,1);  
endfor

r_2 =0;
for i = 1: rows(Theta2)
  for j = 2 : columns(Theta2)
    r_2 = r_2 + Theta2(i,j) *Theta2(i,j);
  endfor
   % take out bias unit in grad
  Theta2_grad(i,1) =  Theta2_grad(i,1)- lambda/m * Theta2(i,1);  
endfor

J = J + lambda/(2*m) *(r_1+r_2);


 


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
