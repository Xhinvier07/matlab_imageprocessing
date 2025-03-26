function [label, confidence] = cnn_classify(imagePath)
    % CNN-based image classification for ginger cats
    % Input: imagePath - path to the image to classify (optional)
    % Output: label - 'Ginger Cat' or 'Not Ginger Cat'
    %         confidence - confidence score (0-100)
    
    % If no image path is provided, open file dialog
    if nargin < 1
        [filename, pathname] = uigetfile({'*.jpg;*.png;*.jpeg;*.bmp', 'Image Files (*.jpg, *.png, *.jpeg, *.bmp)';
                                        '*.*', 'All Files (*.*)'}, ...
                                        'Select an image to classify');
        if filename == 0
            error('No image selected');
        end
        imagePath = fullfile(pathname, filename);
    end
    
    % Load the trained network
    try
        load('ginger_cat_classification/ginger_cat_detector.mat', 'net');
    catch
        error('Could not find the trained network. Please run cnn_train.m first.');
    end
    
    % Read and preprocess the image
    img = imread(imagePath);
    img_resized = imresize(img, [224 224]);  % Resize to match network input size
    
    % Classify the image
    [label, scores] = classify(net, img_resized);
    confidence = max(scores) * 100;
    
    % Display results
    figure('Name', 'Ginger Cat Classification');
    subplot(1,2,1);
    imshow(img);
    title('Original Image');
    
    subplot(1,2,2);
    imshow(img_resized);
    title(sprintf('Classification: %s\nConfidence: %.1f%%', char(label), confidence));
    
    % Add colored border based on classification
    if label == 'Ginger Cat'
        rectangle('Position', [1, 1, size(img_resized, 2)-1, size(img_resized, 1)-1], ...
                  'EdgeColor', 'g', 'LineWidth', 2);
    else
        rectangle('Position', [1, 1, size(img_resized, 2)-1, size(img_resized, 1)-1], ...
                  'EdgeColor', 'b', 'LineWidth', 2);
    end
    
    % Create output folder if it doesn't exist
    outputFolder = 'ginger_cat_classification/classification_test';
    if ~exist(outputFolder, 'dir')
        mkdir(outputFolder);
    end
    
    % Generate filename with timestamp and classification
    [~, originalName, ~] = fileparts(imagePath);
    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    classification = char(label);
    confidenceStr = sprintf('%.1f', confidence);
    outputFilename = sprintf('%s_%s_%s_%s%%', originalName, classification, confidenceStr, timestamp);
    
    % Save the figure
    saveas(gcf, fullfile(outputFolder, [outputFilename '.png']));
    
    % Save the original image with classification info
    classifiedImg = img;
    % Add text to the image
    textStr = sprintf('%s (%.1f%%)', classification, confidence);
    classifiedImg = insertText(classifiedImg, [10 10], textStr, ...
        'FontSize', 20, 'BoxColor', 'white', 'BoxOpacity', 0.7);
    
    % Save the classified image
    imwrite(classifiedImg, fullfile(outputFolder, [outputFilename '_original.png']));
    
    fprintf('Classification results saved in: %s\n', outputFolder);
    fprintf('Files saved as:\n');
    fprintf('- %s.png (figure with both images)\n', outputFilename);
    fprintf('- %s_original.png (original image with classification text)\n', outputFilename);
end

% Example usage:
% [label, confidence] = cnn_classify();  % Opens file dialog
% [label, confidence] = cnn_classify('path/to/your/image.jpg');  % Direct path
