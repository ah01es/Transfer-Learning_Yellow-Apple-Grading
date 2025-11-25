%% Transfer Learning Using AlexNet

close all
clear all
clc

% Get Data
data= '.\DA';
imds = imageDatastore(data,'IncludeSubfolders'...
    ,true, ...
    'LabelSource','foldernames');

% Dividing data into training, validation and test sets
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.8,'randomized');
[imdsTrain,imdsTest] = splitEachLabel(imdsTrain,0.75,'randomized');

% Show smples of training images
numTrainImages = numel(imdsTrain.Labels);
idx = randperm(numTrainImages,16);
figure
for i = 1:16
    subplot(4,4,i)
    I = readimage(imdsTrain,idx(i));
    imshow(I)
end

%Loading the AlexNet network and preparing the layers
net = alexnet;
analyzeNetwork(net)
inputSize = net.Layers(1).InputSize

%Selection of Transfer Learning layers
layersTransfer = net.Layers(1:end-3);

% Define new layers
numClasses = numel(categories(imdsTrain.Labels))
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];

% Creating a DataAugmentation function
pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);

% Creating augmented data sets
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);

augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);
augimdsTest = augmentedImageDatastore(inputSize(1:2),imdsTest);

% Training settings
options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',10, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',3, ...
    'Verbose',false, ...
    'Plots','training-progress');


netTransfer = trainNetwork(augimdsTrain,layers,options);


YPred = classify(netTransfer,augimdsValidation);
YValidation = imdsValidation.Labels;
Validation_accuracy = mean(YPred == YValidation)

C = confusionmat(YValidation, YPred);
figure,confusionchart(C, unique(YValidation));



Ytest = classify(netTransfer,augimdsTest);

Yreal = imdsTest.Labels;
Test_accuracy = mean(Ytest == Yreal)

idx = find(Yreal ~= Ytest);
disp(idx);
j = numel(idx);
k = sqrt(j);
figure
for i = 1:j
    subplot(k,k,i)
    I = readimage(imdsTest,idx(i));
    imshow(I)
    label = ['Yt = ', Ytest(idx(i)), ' & Yr = ', Yreal(idx(i))];
    title(string(label));
end

C = confusionmat(Yreal, Ytest);
figure, confusionchart(C, unique(Yreal));
