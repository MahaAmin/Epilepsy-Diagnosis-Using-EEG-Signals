function project()

C=2;    %number of classes
H=20;   %number of features
train_size=1000;
test_size=1000;
%%%%%%%%%%%%%%%% Bayesian Classifier %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% F --> 1 , N --> 2 , O --> 3 , S --> 4 , Z --> 5 
%%%% XTrain --> (train_size, 21)                                       %%%%
%%%% XTest --> (test_size, 20)                                         %%%%
%%%% YTrain --> (train_size, 1)                                        %%%%
%%%% YTest --> (test_size, 1)                                          %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% call prepareData(H, train_size, test_size)
[ XTrain, YTrain, XTest, YTest ] = prepareData(H, train_size, test_size);

%[ XTrain, YTrain, XTest, YTest ] = prepareData2(H, train_size, test_size);

% Maximum Likelihood Estimation
YPredict = GenericBayesianClassifier(C,H,train_size,XTrain, test_size, XTest );

% Call CalculateAccuracy given YTest and YPredict
accuracy = CalculateAccuracy(YTest, YPredict);
fprintf('\n Generic Bayesian Classifier accuracy =  ');
disp(accuracy);
        

% KNN with Euclidean Distance
YPredict = KNNEuclidean( XTrain, train_size, XTest, test_size, 155 );
% Call CalculateAccuracy given YTest and YPredict
accuracy = CalculateAccuracy(YTest, YPredict);
fprintf('\n KNN with Euclidean distance Classifier accuracy =  ');
disp(accuracy);


% % KNN with Cosine Distance
YPredict = KNNCosine( XTrain, train_size, XTest, test_size, 155 );
% Call CalculateAccuracy given YTest and YPredict
accuracy = CalculateAccuracy(YTest, YPredict);
fprintf('\n KNN with Cosine distance Classifier accuracy =  ');
disp(accuracy);



% Neural Network
YPredict = NN( H, XTrain, train_size, YTrain, XTest, test_size);
% Call CalculateAccuracy given YTest and YPredict
accuracy = CalculateAccuracy(YTest, YPredict);
fprintf('\n Neural Network accuracy =  ');
disp(accuracy);


end

function accuracy = CalculateAccuracy(YTrue, YPredict)
    
    %Hint:: you can get the 1 dimenssion of the vector by size(vec,1)
    sz = size(YTrue, 1);
    
    %Hint:: you can know the true predictions by creating a boolean vector 
    %using YTrue == YPredict;
    check = YTrue == YPredict;
    truePredictions =sum(check);
    %Hint:: accuracy = summation(truePredictions) / #samples
    accuracy = truePredictions / sz;
    accuracy = accuracy*100;
    
end



