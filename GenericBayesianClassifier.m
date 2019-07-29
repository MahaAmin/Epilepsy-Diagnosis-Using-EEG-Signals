function [ YPredict ] = GenericBayesianClassifier(C, H, train_size, XTrain, test_size, XTest )
% PRIORS
Pw = [0.2 0.2 0.2 0.2 0.2];

% Mus --> Call EstimateMus given XTrain
Mus = EstimateMus(H, train_size, XTrain);

% Sigmas --> Call EstimateSigmas given Xtrain and mus
Sigmas = EstimateSigmas(H, train_size, XTrain, Mus );


% Compute the likelihood values for the given features
m = test_size; % number of samples
YPredict=zeros(m,1);
Pw = [0.2 0.2 0.2 0.2 0.2]; %prior
Pxjwi = zeros(C,H); %likelihoods

for k=1:m
    for i=1:C
        for j=1:H
            Pxjwi(i,j) = (1 / sqrt(2 * pi) * Sigmas(i,j)) * exp(-(XTest(k,j) - Mus(i,j))^2/(2 * Sigmas(i,j)^2));
        end
    end

PXw = ones(C,1);   
for i=1:C
    for j=1:(H-1)
        PXw(i) = PXw(i)* Pxjwi(i,j) * Pxjwi(i,j+1);
    end
end
% Compute the posterior probabilities P(wi|x1,x2) by which we decide
% that the given features belongs to which class!
PwX = zeros(C,1); %Posteriors
sum = 0;
for i=1:C
    PwX(i) = PXw(i) * Pw(i);
    sum = sum + PwX(i);
end
PX = sum;

% Normalize the posteriors
for i=1:C
    PwX(i) = PwX(i) / PX;
end
% Decide preidicted class
[~,I] = max(PwX);
YPredict(k,1)=I;
end

end


