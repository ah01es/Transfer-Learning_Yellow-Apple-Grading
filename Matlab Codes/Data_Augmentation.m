close all
clear all
clc

% Set the input and output folder path
inputFolder1 = '..\Images\class1'; % First folder path
inputFolder2 = '..\Images\Class2'; % Second folder path
inputFolder3 = '..\Images\Reject'; % Third folder path
outputFolder1 = '.\DA\1'; % Output folder path
outputFolder2 = '.\DA\2'; % Output folder path
outputFolder3 = '.\DA\3'; % Output folder path

% Number of desired images for each folder (400 images)
desiredNumImages = 400;

% Reading images of each folder
images1 = imageDatastore(inputFolder1);
images2 = imageDatastore(inputFolder2);
images3 = imageDatastore(inputFolder3);

% Calculate the number of images in each folder:
numImages1 = length(images1.Files);
numImages2 = length(images2.Files);
numImages3 = length(images3.Files);

% Number of images to add to each folder
addImages1 = max(0, desiredNumImages - numImages1);
addImages2 = max(0, desiredNumImages - numImages2);
addImages3 = max(0, desiredNumImages - numImages3);

% Set up data augmentation techniques
augmenter = imageDataAugmenter( ...
    'RandRotation', [-10, 10], ...
    'RandXReflection', true, ...
    'RandYReflection', true);

% Generate and save added images
for i = 1:addImages1
    % Random selection of image from the first folder
    img = readimage(images1, randi(numImages1));

    
    % Apply augmentation techniques
    augmentedImg = augment(augmenter, img);
    
    % Save the added image in the output folder
    imwrite(augmentedImg, fullfile(outputFolder1, sprintf('image_%04d.jpg', i)));
end

% Generate and save added images
for i = 1:addImages2
    img = readimage(images2, randi(numImages2));
    augmentedImg = augment(augmenter, img);
    imwrite(augmentedImg, fullfile(outputFolder2, sprintf('image_%04d.jpg', i + addImages1)));
end

% Generate and save added images
for i = 1:addImages3
    img = readimage(images3, randi(numImages3));
    augmentedImg = augment(augmenter, img);
    imwrite(augmentedImg, fullfile(outputFolder3, sprintf('image_%04d.jpg', i + addImages1 + addImages2)));
end

disp('Data Augmentation successfully.');
