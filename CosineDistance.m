function Distances = CosineDistance(X, sample)
%X --> XTrain(train_size, 21)
%sample --> point from XTest sample(1,20)
%Distances(m,2)

% cosine(v1, v2) = (v1.v2)/||v1||*||v2||

m = size(X,1);
Distances = zeros(m,2);
for i=1:m
    dotProduct=dot(sample,X(i,1:20));   %v1.v2
    
    sv=sample .* sample; %vector of sample
    dp=sum(sv);
    magSample=sqrt(dp);  % ||v1||
    
    sv=X(i,1:20) .* X(i,1:20);  %vector of xtrain
    dp=sum(sv);
    magXi=sqrt(dp);     % ||v2||
    
    Distances(i,1)=dotProduct / (magSample*magXi);  % cos(v1, v2)
    Distances(i,2)=X(i,21);     % label_class
end


end

