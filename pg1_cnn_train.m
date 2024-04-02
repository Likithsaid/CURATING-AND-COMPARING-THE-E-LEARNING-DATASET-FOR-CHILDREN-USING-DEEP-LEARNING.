clear all;
close all;
clc;
dDataPth='C:\Engagement Prediction\Children Dataset\Final Dataset\train';
dImages=imageDatastore(dDataPth,'IncludeSubfolders',true,'LabelSource','foldernames');
digitDatasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos', ...
    'nndatasets','DigitDataset');
% 
% dImages = imageDatastore(digitDatasetPath, ...
%     'IncludeSubfolders',true, ...
%     'LabelSource','foldernames');
% dImages = imresize(dImages., [28 28]);

inputSize = [100 100];
dImages.ReadFcn = @(loc)imresize(imread(loc),inputSize);

numTrain=0.85;
[TrImages,TstImages]=splitEachLabel(dImages,numTrain,'randomize');
layers=[
           imageInputLayer([100 100 1],'Name','Input')
           
           convolution2dLayer(3,8,'Padding','same','Name','Conv_1')
           batchNormalizationLayer('Name','BN1')
           reluLayer('Name','Relu_1')
           maxPooling2dLayer(2,'Stride',2,'Name','MaxPool1')
           
          
           convolution2dLayer(3,16,'Padding','same','Name','Conv_2')
           batchNormalizationLayer('Name','BN2')
           reluLayer('Name','Relu_2')
           maxPooling2dLayer(2,'Stride',2,'Name','MaxPool2')
           
           
           convolution2dLayer(3,32,'Padding','same','Name','Conv_3')
           batchNormalizationLayer('Name','BN3')
           reluLayer('Name','Relu_3')
           maxPooling2dLayer(2,'Stride',2,'Name','MaxPool3')
           
           convolution2dLayer(3,64,'Padding','same','Name','Conv_4')
           batchNormalizationLayer('Name','BN4')
           reluLayer('Name','Relu_4')           
            maxPooling2dLayer(2,'Stride',2,'Name','MaxPool4')
            
            
           convolution2dLayer(3,64,'Padding','same','Name','Conv_5')
           batchNormalizationLayer('Name','BN5')
           reluLayer('Name','Relu_5')
           maxPooling2dLayer(2,'Stride',2,'Name','MaxPool5')
           
           convolution2dLayer(3,128,'Padding','same','Name','Conv_6')
           batchNormalizationLayer('Name','BN6')
           reluLayer('Name','Relu_6')
%            maxPooling2dLayer(2,'Stride',2,'Name','MaxPool4')
           
           fullyConnectedLayer(3,'Name','FC')
           softmaxLayer('Name','softmax')
           classificationLayer('Name','OutClass')
           ];
       lgrph=layerGraph(layers);
       plot(lgrph);
       opt=trainingOptions('sgdm','initialLearnRate', 0.01,'MaxEpochs',10,'Shuffle',...
       'every-epoch','ValidationData',TstImages,'ValidationFrequency',10,'Verbose',false,...
       'Plots','training-progress');
   net=trainNetwork(TrImages,layers,opt);
   YPred=classify(net,TstImages);
   YValid=TstImages.Labels;
   accrcy=sum(YPred==YValid)/numel(YValid)
   
   save("eng1.mat");
   
       
           
           
           