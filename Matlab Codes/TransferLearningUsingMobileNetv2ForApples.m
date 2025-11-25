close all
clear
clc

% Get Data
data= '.\DA';
imds = imageDatastore(data, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% Dividing data into training, validation and test sets
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.8,'randomized');
[imdsTrain,imdsTest] = splitEachLabel(imdsTrain,0.75,'randomized');

% Set standard size for resizing images
inputSize = [224 224 3];

% Determining the range of pixels to move images
pixelRange = [-30 30];

% Creating a data increment function
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);

% Creating augmented data sets
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation); % inputSize = [224 224 3] =>> 224x224
augimdsTest = augmentedImageDatastore(inputSize(1:2),imdsTest);



% Load the MobileNetV2 model and convert it to a LayerGraph
net = mobilenetv2;
lgraph = layerGraph(net);

% Find learnable layers and classification layer
[learnableLayer,classLayer] = findLayersToReplace(lgraph);

%Create and replace a new layer
numClasses = 3;
newLearnableLayer = fullyConnectedLayer(numClasses,'Name', 'predictions');
lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);

newClassLayer = classificationLayer('Name', 'ClassificationLayer_predictions');
lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);

% Training settings
options = trainingOptions("rmsprop", ...
    'MaxEpochs',10, ...
    'MiniBatchSize',10, ...
    'Shuffle','every-epoch', ...
    'ValidationData', augimdsValidation, ...
    'ValidationFrequency',3, ...
    'InitialLearnRate',1e-4, ...
    'SquaredGradientDecayFactor',0.99, ...
    'ResetInputNormalization',false, ...
    'Plots','training-progress');

% Network training
net = trainNetwork(augimdsTrain, lgraph, options);

% Network testing and accuracy calculation
YPred = classify(net, augimdsTest);
Yreal = imdsTest.Labels;
accuracy = sum(YPred == Yreal) / numel(Yreal);
fprintf('Accuracy = %.2f%%\n', accuracy * 100);

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

% Create and display confusion matrix
C = confusionmat(Yreal, YPred);
figure, confusionchart(C, unique(Yreal));
