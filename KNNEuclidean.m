function [ YPredict ] = KNNEuclidean( XTrain, train_size, XTest, test_size, K )

YPredict=zeros(test_size,1);
for i=1:test_size
    % Step#1 compute Ecludiean distance between sample point and each point in
    % the training set
    Distances=EuclideanDistance(XTrain, XTest(i,:)); %Distances(train_size,2)
    %Distance(i,1) --> Distance value
    %Distance(i,2) --> class label
    
    
    % Step#2 Sort distances in ascending order
    Distances=sortrows(Distances,1);

    % Step#3 Select the first K-patterns
    % Step#4 For each class count the number of points from the selected
    % K-patterns --> Ki
    KI=zeros(2,1);  % 2 classes 1 count
    for l=1:K
        if(Distances(l,2)==0)
            KI(1,1)=KI(1,1)+1;
        elseif(Distances(l,2)==1)
            KI(2,1)=KI(2,1)+1;
        end
    end

    % Step#5 P(wi|x) = Ki/K
    PwiX=zeros(2,1);
    PwiX(1,1)=KI(1,1)/K;
    PwiX(2,1)=KI(2,1)/K;
    
    % Choose which class the sample point belongs to
    if(PwiX(1,1)>=PwiX(2,1))
        YPredict(i,1)=0;
    else
        YPredict(i,1)=1;
    end
end

end

