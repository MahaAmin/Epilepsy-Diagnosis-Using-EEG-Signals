function Distances = EuclideanDistance( X, sample )

%X --> XTrain(train_size, 21)
%sample --> point from XTest sample(1,20)
%Distances(m,2)

m = size(X,1);
Distances = zeros(m,2);
for i=1:m
    sum=0;
    for j=1:20
        sum=sum+((sample(1,j)-X(i,j))^2);
    end
    Distances(i,1)=sqrt(sum);
    Distances(i,2)=X(i,21);
end

end

