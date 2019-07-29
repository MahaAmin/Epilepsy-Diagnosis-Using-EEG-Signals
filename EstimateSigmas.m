function Sigmas = EstimateSigmas(H, train_size, XTrain, Mus )

% H --> number of features
% XTrain (train_size, H+1)
% Mus (2,H)

Sigmas=zeros(2,H);
Features=zeros(train_size,H,2);

%ubnormal_class
ubnormal=(train_size/5)*3;
Features(1:ubnormal,:,1)= XTrain(1:ubnormal,1:20)-Mus(1,:);
Sigmas(1,:)=sqrt(sum(Features(1:ubnormal,:,1).^2)./ubnormal);


%normal_class
normal=(train_size/5)*2;
Features(ubnormal+1:end,:,2)= XTrain(ubnormal+1:end,1:20)-Mus(2,:);
Sigmas(2,:)=sqrt(sum(Features(ubnormal+1:end,:,2).^2)./normal);


end

