function [ XTrain, YTrain, XTest, YTest ] = prepareData(H, train_size, test_size)
    F=readtable('F.txt');
    F=F{:,:}; %upnormal
    N=readtable('N.txt');
    N=N{:,:};   %upnormal
    S=readtable('S.txt');
    S=S{:,:};   %upnormal
    O=readtable('O.txt');
    O=O{:,:};   %normal
    Z=readtable('z.txt');
    Z=Z{:,:};   %normal
    
    
    %add column indicating class label (normal=1 --> Z & O, 
    %upnormal=0 --> N & F & S)
    upnormalLabel = zeros(400,1);
    normalLabel = ones(400,1);
    
    F = [F upnormalLabel];
    N = [N upnormalLabel];
    S = [S upnormalLabel];
    O = [O normalLabel];
    Z = [Z normalLabel];
    
    %Merge data in one matrix data(2000,21)
    data=[F;N;S;O;Z];
    
    %normalize data(all values [0 1]
    for i=1:20
        data(:,i)=data(:,i)/max(data(:,i));
    end
    
    
    %Prepare XTrain, YTrain
    XTrain=zeros(train_size,H+1);
    YTrain=zeros(train_size,1);
    
    %XTrain & YTrain with shuffling (train_size/5) rows from each class 
    %XTrain(500,21), YTrain(500,1)
    trainFromEachClass = train_size/5;  %500/5=100
    testFromEcahClass = test_size/5;    %1500/5=300
    for i=1:5
        XTrain((i-1)*trainFromEachClass+1:i*(train_size/5),:)= data((i-1)*400+1:(i-1)*400+trainFromEachClass,:);
        YTrain((i-1)*trainFromEachClass+1:i*(train_size/5),:)= data((i-1)*400+1:(i-1)*400+trainFromEachClass,21);       
    end
    
    
    %Prepare XTest & YTest
    XTest=zeros(test_size,H);
    YTest=zeros(test_size,1);
    
    %XTest & YTest with shuffling (test_size/5) rows from each class
    for i=1:5
        XTest((i-1)*testFromEcahClass+1:i*(test_size/5),:)=data((i-1)*400+trainFromEachClass+1:(i-1)*400+trainFromEachClass+testFromEcahClass,1:20);
        YTest((i-1)*testFromEcahClass+1:i*(test_size/5),1)=data((i-1)*400+trainFromEachClass+1:(i-1)*400+trainFromEachClass+testFromEcahClass,21);
    end
        

end

