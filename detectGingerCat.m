% CNN-based image classification script for ginger cats
% Input folder: images/processed_images
% Output folder: ginger_cat_classification

% Setup folders
inputFolder = 'images/gingercat_images';
outputFolder = 'ginger_cat_classification';

% Create output folder if it doesn't exist
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

% Define categories for binary classification
categories = {'ginger_cat', 'not_ginger_cat'};

% Image input size for the network
inputSize = [224 224 3];  % Standard size for many CNNs

% Create image datastore
% Since we don't have folders by category, we'll need to label manually
imageFiles = dir(fullfile(inputFolder, '*.jpg'));
labels = cell(length(imageFiles), 1);

% Label images based on content
filePaths = cell(length(imageFiles), 1);
for i = 1:length(imageFiles)
    filePath = fullfile(inputFolder, imageFiles(i).name);
    filePaths{i} = filePath;
    
    % Read image
    img = imread(filePath);
    
    % Determine if it's a ginger cat (using the function defined below)
    if detectGingerCat(img)
        labels{i} = 'ginger_cat';
    else
        labels{i} = 'not_ginger_cat';
    end
end

% Create datastore with manual labels
imds = imageDatastore(filePaths, 'Labels', categorical(labels));

% Display class distribution
numGingerCat = sum(imds.Labels == 'ginger_cat');
numNotGingerCat = sum(imds.Labels == 'not_ginger_cat');
fprintf('Class distribution:\n');
fprintf('- Ginger cat: %d images\n', numGingerCat);
fprintf('- Not ginger cat: %d images\n', numNotGingerCat);

% Create data augmenter to increase training data variety
augmenter = imageDataAugmenter('RandRotation', [-20, 20], ...
    'RandXTranslation', [-10 10], ...
    'RandYTranslation', [-10 10], ...
    'RandXScale', [0.8 1.2], ...
    'RandYScale', [0.8 1.2]);

% Define CNN Architecture - simplified for binary classification
layers = [
    % Input layer
    imageInputLayer(inputSize)
    
    % First convolution block
    convolution2dLayer(3, 16, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    
    % Second convolution block
    convolution2dLayer(3, 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    
    % Third convolution block
    convolution2dLayer(3, 64, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    
    % Fully connected layers
    fullyConnectedLayer(128)
    reluLayer
    dropoutLayer(0.5)
    
    % Output layer - binary classification
    fullyConnectedLayer(2)  % Two classes: ginger_cat and not_ginger_cat
    softmaxLayer
    classificationLayer
];

% Training options - fewer epochs for binary classification
options = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.001, ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 8, ...
    'Shuffle', 'every-epoch', ...
    'ValidationFrequency', 10, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'auto');

% Train the network
try
    % Split data into training and validation sets
    [trainingSet, validationSet] = splitEachLabel(imds, 0.7, 'randomized');
    
    % Create augmented image datastores for training and validation
    augTrainingSet = augmentedImageDatastore(inputSize, trainingSet, ...
        'DataAugmentation', augmenter);
    augValidationSet = augmentedImageDatastore(inputSize, validationSet);
    
    % Train the network
    net = trainNetwork(augTrainingSet, layers, options);
    
    % Save the trained network
    save(fullfile(outputFolder, 'ginger_cat_detector.mat'), 'net');
    
    % Evaluate network performance
    YPred = classify(net, augValidationSet);
    YValidation = validationSet.Labels;
    accuracy = sum(YPred == YValidation)/numel(YValidation);
    
    % Create confusion matrix
    figure('Name', 'Classification Results');
    cm = confusionmat(YValidation, YPred);
    confusionchart(cm, categories);
    title(['Confusion Matrix (Accuracy: ' num2str(accuracy*100) '%)']);
    saveas(gcf, fullfile(outputFolder, 'confusion_matrix.png'));
    
    % Calculate more detailed metrics
    tp = cm(1,1); % true positive
    fn = cm(1,2); % false negative
    fp = cm(2,1); % false positive
    tn = cm(2,2); % true negative
    
    precision = tp / (tp + fp);
    recall = tp / (tp + fn);
    f1_score = 2 * (precision * recall) / (precision + recall);
    
    % Test on individual images
    figure('Name', 'Ginger Cat Detection Results');
    for i = 1:min(8, length(imds.Files)) % Show up to 8 images
        % Read and preprocess image
        img = readimage(imds, i);
        img_resized = imresize(img, inputSize(1:2));
        
        % Classify image
        [label, scores] = classify(net, img_resized);
        
        % Display results
        subplot(2, 4, i);
        imshow(img);
        
        % Add colored border based on classification
        if label == 'ginger_cat'
            if imds.Labels(i) == 'ginger_cat'
                % True positive - green border
                rectangle('Position', [1, 1, size(img, 2)-1, size(img, 1)-1], ...
                          'EdgeColor', 'g', 'LineWidth', 3);
            else
                % False positive - yellow border
                rectangle('Position', [1, 1, size(img, 2)-1, size(img, 1)-1], ...
                          'EdgeColor', 'y', 'LineWidth', 3);
            end
        else
            if imds.Labels(i) == 'not_ginger_cat'
                % True negative - blue border
                rectangle('Position', [1, 1, size(img, 2)-1, size(img, 1)-1], ...
                          'EdgeColor', 'b', 'LineWidth', 3);
            else
                % False negative - red border
                rectangle('Position', [1, 1, size(img, 2)-1, size(img, 1)-1], ...
                          'EdgeColor', 'r', 'LineWidth', 3);
            end
        end
        
        confidence = max(scores) * 100;
        title({char(label), ['Conf: ' num2str(confidence, '%.1f') '%']}, 'FontSize', 8);
    end
    saveas(gcf, fullfile(outputFolder, 'classification_results.png'));
    
    % Save classification metrics to file
    metrics_file = fullfile(outputFolder, 'classification_metrics.txt');
    fid = fopen(metrics_file, 'w');
    fprintf(fid, 'Ginger Cat Classification Metrics\n');
    fprintf(fid, '--------------------------------\n\n');
    fprintf(fid, 'Overall Accuracy: %.2f%%\n\n', accuracy*100);
    fprintf(fid, 'Precision: %.2f%%\n', precision*100);
    fprintf(fid, 'Recall: %.2f%%\n', recall*100);
    fprintf(fid, 'F1 Score: %.2f%%\n\n', f1_score*100);
    fprintf(fid, 'Confusion Matrix:\n');
    fprintf(fid, '             Predicted      \n');
    fprintf(fid, '              Cat    Not Cat\n');
    fprintf(fid, 'Actual Cat    %4d    %4d\n', tp, fn);
    fprintf(fid, 'Actual Not    %4d    %4d\n', fp, tn);
    fclose(fid);
    
catch ME
    fprintf('Error during training: %s\n', ME.message);
    fprintf('Please ensure you have:\n');
    fprintf('1. Deep Learning Toolbox installed\n');
    fprintf('2. Sufficient images for training\n');
    fprintf('3. At least some images that contain ginger cats\n');
end

fprintf('\nProcessing complete! Results saved in %s\n', outputFolder);
fprintf('The model is trained to detect ginger cats versus other images.\n');
fprintf('Network architecture:\n');
fprintf('- Input size: 224x224x3\n');
fprintf('- 3 Convolution blocks with increasing filters (16, 32, 64)\n');
fprintf('- Binary classification output (ginger_cat vs. not_ginger_cat)\n');