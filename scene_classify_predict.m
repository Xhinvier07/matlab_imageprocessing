% Load trained KNN model
load('scene_classification/knn_model.mat');

% Read new image
new_img = imread('new_image.jpg');

% Extract features
new_features = extractColorHistogram(new_img);
new_features = [new_features extractTextureFeatures(new_img)];

% Predict using KNN
predicted_label = predict(knn_model, new_features);
disp(['Predicted Class: ', char(predicted_label)]);
