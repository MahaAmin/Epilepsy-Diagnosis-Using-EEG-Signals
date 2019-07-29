function YPredict = NN( H, XTrain, train_size, YTrain, XTest, test_size)
% H = 20 % number of features
% XTrain (train_size, 21)
% XTest (test_size, 20)
% YTest (test_size, 1)
alpha = 0.1;       %learning rate
input_layer = H;    % no. of neurons in input_layer = number of features
hidden_layer = 2;   % no. of neurons in hidden layer
output_layer = 2;   % no. of neurons in output_layer = number of classes
epochs=50;
costs=zeros(epochs,2);

% Step#0 initialize thetas randomly
theta1=rand(input_layer+1, hidden_layer);   % theta1 = (21, 2)
theta2=rand(hidden_layer+1, output_layer);  % theta2 = (3, 2)


%################# Training ###############################################
for e=1:epochs
    % Step#1 apply feedforward propagation
    [h, z2, a0, a1] = feedforward(XTrain(:,1:20),train_size,theta1, theta2);
    
    % Step#2 calculate cost
    logCost=logisticCost(h, train_size, YTrain);
    costs(e,:)=logCost;
    
    % Step#3 apply back-propagation
    [delta1, delta2] = backpropagation(h, train_size, YTrain, theta2, z2, a0, a1);
    
    
    % Step#4 update weights using gradient descent (gradients, alpha)
    [theta1, theta2] = gradientDescent(train_size, delta1, delta2, a0, a1, theta1, theta2, alpha);
end


plot(costs);
%#################### Testing #############################################
[h, ~, ~, ~] = feedforward(XTest,test_size, theta1, theta2);

YPredict=zeros(test_size,1);
for i=1:test_size
    if(h(i,1)>=h(i,2))
        YPredict(i)=0;
    elseif(h(i,2)>h(i,1))
        YPredict(i)=1;
    end
end


end

function [h, z2, a0, a1] = feedforward(XTrain,train_size, theta1, theta2)
a0 = [ones(train_size,1) XTrain(:,:)];   % a0 = (train_size, 21) 21=bias+features
z1 = a0 * theta1;   % theta1 = (21, 2) --> z1 = (train_size, 2)
gz1 = Sigmoid(z1);      % g(z1) = sigmoid(g(z1))
a1 = [ones(train_size,1) gz1]; % a1 = (train_size, 3)
z2 = a1*theta2;   % 
a2 = Sigmoid(z2); 
h = a2;     % hypothesis

end


function [delta1, delta2]=backpropagation(Hypothesis, train_size, YTrain, theta2, z, a0, a1)
delta2=Hypothesis-YTrain;
delta1=delta2*transpose(theta2);
sigmoidGradient=gd([ones(train_size,1) z]);
delta1=delta1 .* sigmoidGradient;
delta1=delta1(:,2:end);
end


function [theta1, theta2] = gradientDescent(train_size, delta1, delta2, a0, a1, theta1, theta2, alpha)
errorRate1=(transpose(delta1)*a0)./train_size;
errorRate1=transpose(errorRate1);
errorRate2=transpose(delta2)*a1;
errorRate2=transpose(errorRate2);
theta1=theta1-(alpha*errorRate1);
theta2=theta2-(alpha*errorRate2);
end


function sigmoid = Sigmoid(z)
sigmoid =1.0 ./ (1.0+exp(-z));
end


function sigmoidGradient = gd(gz)
sigmoidGradient=gz .* (1-gz);
end


function logCost=logisticCost(Hypothesis, train_size, YTrain)
logCost=(1/train_size)*sum(-YTrain.*log(Hypothesis) - (1-YTrain).*log(1-Hypothesis));
end

