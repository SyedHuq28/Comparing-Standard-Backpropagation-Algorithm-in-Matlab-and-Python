% Read the csv file where data is already separated, normalized and feature and target values
% are separate and denoted as 'X (1).csv' and 'y (1).csv'
X = readtable('X (1).csv');
X = table2array(X);
Y = readtable('y (1).csv');
Y = table2array(Y);

% Split dataset into training and testing sets
n = length(Y);
hpartition = cvpartition(n,'Holdout',0.3);
idxTrain = training(hpartition);
trainX = X(idxTrain,:);
trainY = Y(idxTrain,:);
idxNew = test(hpartition);
testX = X(idxNew,:);
testY = Y(idxNew,:);

%create neural network layers
input_layer_size = 11;
hidden_layer_size = 21;
output_layer_size = 1;

%Create random weights
W1 = 2*rand(input_layer_size, hidden_layer_size) - 1;
W2 = 2*rand(hidden_layer_size, output_layer_size) - 1;

%Specify number of epochs and learning rate
epochs = 100;
learning_rate = 0.1;

%Check the loss value of test set before implementing algorithm
[a2,a3] = forward_propagation(testX, W1, W2);
test_loss = mean((a3 - testY).^2);

%Implementing standard backpropagation algorithm with single hidden layer
%Recording the time taken for the algorithm to complete

time_elapsed = 0;
for i = 1:epochs
    tic
    % Forward propagation
    [a2,a3] = forward_propagation(trainX, W1, W2);
    
    %Calculate training loss at each epoch before backward propagation
    train_loss = mean((a3 - trainY).^2);
    train_errors(i) = train_loss;
    fprintf('Epoch %d, train loss = %.4f\n', i, train_loss);
    
    %Calulcate accuracy of each epoch
    accuracy = mean(int64(a3) == int64(trainY))*100;
    accuracy_increase(i) = accuracy;
    fprintf('Accuracy = %.4f\n', accuracy);
 
    % Backward propagation
    [back1, back2] = backward_propagation(trainX, trainY, W2, a2, a3);

    % Updating weights through backward propagation
    W1 = W1 - learning_rate * back1;
    W2 = W2 - learning_rate * back2;

    %Calulating time taken to run all epoch
    time_elapsed = time_elapsed + toc;
    time_increase(i) = time_elapsed;

end
fprintf('Time taken = %.4f\n', time_elapsed);


figure(1);
%PLotting a simple linear graph showing training loss at each epoch
iterations = 1:epochs;
plot(iterations, train_errors);
title('Improvement of Training Loss');
xlabel('No. of Epochs');
ylabel('Loss Value');
legend('Training loss');

figure(2);
plot(iterations, accuracy_increase);
title('Improvement of Accuracy');
xlabel('No. of Epochs');
ylabel('Accuracy %');
legend('Accuracy');

figure(3);
plot(iterations, time_increase);
title('Time elapsed');
xlabel('No. of Epochs');
ylabel('Time in seconds');
%legend('Time');

%Function for forward propagation
function [a2,a3] = forward_propagation(X, W1, W2)
    z2 = X * W1;
    a2 = sigmoid(z2);
    z3 = a2 * W2;
    a3 = sigmoid(z3);
end

%Function for backward propagation
function [back1, back2] = backward_propagation(X, Y, W2, a2, a3)
    delta1 = (a3 - Y) .* sigmoid_deriv(a3);
    delta2 = (delta1 * W2') .* sigmoid_deriv(a2);
    back2 = a2' * delta1;
    back1 = X' * delta2;
end

%Sigmoid function for forward propagation
function [sigmoid] = sigmoid(z)
    sigmoid = 1./(1+exp(-z));
end

%Getting the derivative of the sigmoid function
function [sigmoid_deriv] = sigmoid_deriv(sigmoid)
    sigmoid_deriv = sigmoid .* (1-sigmoid);
end

