% Load trained KNN model
load('scene_classification/knn_model.mat');

% Select an image file
[file, path] = uigetfile({'*.jpg;*.jpeg;*.png;*.bmp;*.tiff', 'All Image Files (*.jpg, *.jpeg, *.png, *.bmp, *.tiff)'; '*.*', 'All Files (*.*)'}, 'Select an Image');
if isequal(file, 0)
    disp('No file selected. Exiting...');
    return;
end

% Read selected image
new_img = imread(fullfile(path, file));

% Extract features using the SAME METHOD as in training
new_features = extractColorHistogram(new_img);
new_features = [new_features extractTextureFeatures(new_img)];

% Ensure feature vector size matches training data
expectedFeatureSize = size(knn_model.X, 2); % Get expected feature size
actualFeatureSize = length(new_features);

if actualFeatureSize ~= expectedFeatureSize
    error('Feature size mismatch! Expected %d, but got %d. Check feature extraction.', expectedFeatureSize, actualFeatureSize);
end

% Classify using KNN
[predicted_label, score] = predict(knn_model, new_features);
confidence = max(score) * 100; % Convert to percentage

% Create a figure to display the image and classification result
fig = figure('Position', [100, 100, 600, 400]);
subplot(1, 2, 1);
imshow(new_img);
title('Selected Image');

% Create text for classification result
pred_text = sprintf('Predicted Class: %s\nConfidence: %.2f%%', char(predicted_label), confidence);

subplot(1, 2, 2);
set(gca, 'Color', 'w');
axis off; % Turn off the axis
text(0.5, 0.5, pred_text, 'HorizontalAlignment', 'center', 'FontSize', 12);

% Save the figure in the scene_classification folder
output_file = fullfile('scene_classification', 'classification_result.png');
saveas(fig, output_file);

% Notify user of save success
disp(['Classification result saved to: ', output_file]);

% ==================== FEATURE EXTRACTION FUNCTIONS ====================

% Function to extract color histogram features (MUST match training)
function hist_features = extractColorHistogram(img)
    hsv_img = rgb2hsv(img);
    num_bins = 16; % MUST be the same as training
    h_hist = histcounts(hsv_img(:,:,1), num_bins, 'Normalization', 'probability');
    s_hist = histcounts(hsv_img(:,:,2), num_bins, 'Normalization', 'probability');
    v_hist = histcounts(hsv_img(:,:,3), num_bins, 'Normalization', 'probability');
    hist_features = [h_hist s_hist v_hist];
end

% Function to extract GLCM texture features (MUST match training)
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