function Mus = EstimateMus(H, train_size, XTrain)

% H --> number of features
% XTrain (train_size, H+1)

Mus = zeros(2,H);

%ubnormal_class
ubnormal=(train_size/5)*3;
Mus(1,:) = sum(XTrain(1:ubnormal,1:20));   
Mus(1,:) =Mus(1,:)/ubnormal;

%normal_class
normal=(train_size/5)*2;
Mus(2,:) = sum(XTrain(ubnormal+1:end,1:20));   
Mus(2,:) =Mus(2,:)/normal;


end

