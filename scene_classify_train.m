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
    'Blue Bird', 'bluebird_images', ...
    'Ginger Cat', 'gingercat_images', ...
    'Strawberry', 'strawberry_images', ...
    'Tree', 'tree_images', ...
    'Rainy Cloud', 'rainy_images', ...
    'Lego Blocks', 'lego_images' ...
);

% Initialize arrays for features and labels
features = [];
labels = [];

% Function to extract color histogram features
function hist_features = extractColorHistogram(img)
    hsv_img = rgb2hsv(img);
    num_bins = 32;
    h_hist = histcounts(hsv_img(:,:,1), num_bins, 'Normalization', 'probability');
    s_hist = histcounts(hsv_img(:,:,2), num_bins, 'Normalization', 'probability');
    v_hist = histcounts(hsv_img(:,:,3), num_bins, 'Normalization', 'probability');
    hist_features = [h_hist s_hist v_hist];
end

% Function to extract GLCM texture features
function texture_features = extractTextureFeatures(img)
    if size(img, 3) == 3
        gray_img = rgb2gray(img);
    else
        gray_img = img;
    end
    offsets = [0 1; -1 1; -1 0; -1 -1];
    glcm = graycomatrix(gray_img, 'Offset', offsets, 'Symmetric', true, 'NumLevels', 8);
    stats = graycoprops(glcm, {'Contrast', 'Correlation', 'Energy', 'Homogeneity'});
    texture_features = [stats.Contrast stats.Correlation stats.Energy stats.Homogeneity];
end

% Loop through predefined categories
classNames = fieldnames(classFolders);
for c = 1:length(classNames)
    className = classNames{c};  % Class label (e.g., 'Blue Bird')
    folderName = classFolders.(className);  % Folder name (e.g., 'bluebird_images')
    categoryPath = fullfile(datasetFolder, folderName);
    
    % Get all image files in this category folder
    if ~exist(categoryPath, 'dir')
        fprintf('Warning: Folder "%s" does not exist, skipping...\n', categoryPath);
        continue;
    end
    imageFiles = dir(fullfile(categoryPath, '*.jpg'));  % Change extension if needed
    
    for i = 1:length(imageFiles)
        img = imread(fullfile(categoryPath, imageFiles(i).name));
        
        % Extract features
        color_features = extractColorHistogram(img);
        texture_features = extractTextureFeatures(img);
        
        % Store extracted features and corresponding label
        features = [features; [color_features texture_features]];
        labels = [labels; {className}];  % Use predefined class label
    end
end

% Convert labels to categorical
labels = categorical(labels);

% Split dataset (70% training, 30% testing)
train_ratio = 0.7;
num_train = round(train_ratio * length(labels));

train_features = features(1:num_train, :);
train_labels = labels(1:num_train);

test_features = features(num_train+1:end, :);
test_labels = labels(num_train+1:end);

% Train KNN classifier
knn_model = fitcknn(train_features, train_labels, 'NumNeighbors', 5);

% Predict using KNN
predicted_labels = predict(knn_model, test_features);

% Calculate accuracy
accuracy = sum(predicted_labels == test_labels) / length(test_labels);
fprintf('KNN Classification Accuracy: %.2f%%\n', accuracy * 100);

% Save trained model
save(fullfile(outputFolder, 'knn_model.mat'), 'knn_model');

% Display confusion matrix
figure;
cm = confusionmat(test_labels, predicted_labels);
confusionchart(cm, classNames);
title(['Confusion Matrix (KNN Accuracy: ' num2str(accuracy*100) '%)']);
saveas(gcf, fullfile(outputFolder, 'confusion_matrix.png'));

fprintf('Processing complete! KNN model saved in %s\n', outputFolder);
