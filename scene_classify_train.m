clc; clear; close all;

% Setup dataset folder
datasetFolder = 'images';  % Root folder containing category-specific subfolders
outputFolder = 'scene_classification';

% Create output folder if it doesn't exist
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

% Define fixed class labels and corresponding subfolders
classFolders = struct( ...
    'Blue_Bird', 'bluebird_images', ...
    'Ginger_Cat', 'gingercat_images_knn', ...
    'Strawberry', 'strawberry_images', ...
    'Tree', 'tree_images', ...
    'Rainy_Cloud', 'rainy_images', ...
    'Lego_Blocks', 'lego_images' ...
);

% Initialize arrays for features and labels
features = [];
labels = [];

% Function to extract color histogram features
function hist_features = extractColorHistogram(img)
    hsv_img = rgb2hsv(img);
    num_bins = 16; % Reduce bins to avoid overfitting
    h_hist = histcounts(hsv_img(:,:,1), num_bins, 'Normalization', 'probability');
    s_hist = histcounts(hsv_img(:,:,2), num_bins, 'Normalization', 'probability');
    v_hist = histcounts(hsv_img(:,:,3), num_bins, 'Normalization', 'probability');
    hist_features = [h_hist s_hist v_hist];
end

% Function to extract GLCM texture features
function texture_features = extractTextureFeatures(img)
    gray_img = rgb2gray(img);
    offsets = [0 1; -1 1; -1 0; -1 -1];
    glcm = graycomatrix(gray_img, 'Offset', offsets, 'Symmetric', true, 'NumLevels', 8);
    stats = graycoprops(glcm, {'Contrast', 'Correlation', 'Energy', 'Homogeneity'});
    texture_features = [stats.Contrast stats.Correlation stats.Energy stats.Homogeneity];
end

% Function for Data Augmentation (Flipping & Rotation)
function img_aug = augmentImage(img)
    methods = {@(x) x, ... % Original
               @(x) imrotate(x, randi([0, 360])), ... % Rotate randomly
               @(x) flip(x, 1), ... % Flip vertically
               @(x) flip(x, 2)}; % Flip horizontally
    idx = randi(length(methods)); 
    img_aug = methods{idx}(img);
end

% Supported image extensions
validExtensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'};

% Loop through predefined categories
classNames = fieldnames(classFolders);
for c = 1:length(classNames)
    className = classNames{c};  
    folderName = classFolders.(className);  
    categoryPath = fullfile(datasetFolder, folderName);
    
    if ~exist(categoryPath, 'dir')
        fprintf('Warning: Folder "%s" does not exist, skipping...\n', categoryPath);
        continue;
    end
    
    % Get all image files with valid extensions
    allFiles = dir(fullfile(categoryPath, '*'));
    imageFiles = allFiles(arrayfun(@(x) any(endsWith(lower(x.name), validExtensions)), allFiles));

    for i = 1:length(imageFiles)
        img = imread(fullfile(categoryPath, imageFiles(i).name));
        
        % Original Image Features
        color_features = extractColorHistogram(img);
        texture_features = extractTextureFeatures(img);
        features = [features; [color_features texture_features]];
        labels = [labels; {className}];
        
        % Augment and Extract Features Again
        img_aug = augmentImage(img);
        color_features_aug = extractColorHistogram(img_aug);
        texture_features_aug = extractTextureFeatures(img_aug);
        features = [features; [color_features_aug texture_features_aug]];
        labels = [labels; {className}];
    end
end

% Convert labels to categorical
labels = categorical(labels);

% Normalize Features
features = normalize(features);

% Stratified Train-Test Split (50% Train, 50% Test)
cv = cvpartition(labels, 'HoldOut', 0.5);
train_features = features(training(cv), :);
train_labels = labels(training(cv));
test_features = features(test(cv), :);
test_labels = labels(test(cv));

% Train KNN classifier with Auto-tuned 'k'
knn_model = fitcknn(train_features, train_labels, 'NumNeighbors', 3, ...
                    'Standardize', true, 'Distance', 'euclidean');

% Predict using KNN
predicted_labels = predict(knn_model, test_features);

% Calculate Accuracy
accuracy = sum(predicted_labels == test_labels) / length(test_labels);
fprintf('KNN Classification Accuracy: %.2f%%\n', accuracy * 100);

% Save Trained Model
save(fullfile(outputFolder, 'knn_model.mat'), 'knn_model');

% Display Confusion Matrix
figure;
cm = confusionmat(test_labels, predicted_labels);
confusionchart(cm, classNames);
title(['Confusion Matrix (KNN Accuracy: ' num2str(accuracy * 100) '%)']);
saveas(gcf, fullfile(outputFolder, 'confusion_matrix.png'));

fprintf('Processing complete! KNN model saved in %s\n', outputFolder);