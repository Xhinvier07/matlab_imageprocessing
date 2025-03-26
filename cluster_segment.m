% Clustering-based segmentation script using K-means
% Input folder: images/processed_images
% Output folder: cluster_segmentation

% Get input from user for number of images to process
num_images = input('Enter the number of images to process: ');
k_clusters = input('Enter the number of clusters (k) for segmentation (recommended 3-5): ');

% Setup folders
inputFolder = 'images/processed_images';
outputFolder = 'cluster_segmentation';

% Create output folder if it doesn't exist
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

% Get list of image files
imageFiles = dir(fullfile(inputFolder, '*.jpg')); % Adjust file extension if needed

% Validate user input
num_images = min(num_images, length(imageFiles));
fprintf('Processing %d images with %d clusters...\n', num_images, k_clusters);

% Process each image
for i = 1:num_images
    % Read image
    img = imread(fullfile(inputFolder, imageFiles(i).name));
    
    % Convert to Lab color space for better color segmentation
    lab_img = rgb2lab(img);
    
    % Reshape the image for clustering
    [height, width, ~] = size(img);
    lab_pixels = reshape(lab_img, height * width, 3);
    
    % Perform K-means clustering
    [cluster_idx, cluster_centers] = kmeans(lab_pixels, k_clusters, ...
        'Distance', 'sqeuclidean', ...
        'Replicates', 3);
    
    % Reshape cluster indices back to image dimensions
    segmented_img = reshape(cluster_idx, height, width);
    
    % Create separate binary masks for each cluster
    cluster_masks = cell(k_clusters, 1);
    cleaned_masks = cell(k_clusters, 1);
    for k = 1:k_clusters
        % Create binary mask for current cluster
        cluster_masks{k} = (segmented_img == k);
        
        % Remove small objects (noise reduction)
        cleaned_masks{k} = bwareaopen(cluster_masks{k}, 100); % Adjust threshold as needed
    end
    
    % Create colored segmentation result
    % Assign different colors to different clusters
    colors = [1 0 0;    % Red
              0 1 0;    % Green
              0 0 1;    % Blue
              1 1 0;    % Yellow
              1 0 1;    % Magenta
              0 1 1;    % Cyan
              0.5 0.5 0.5]; % Gray
    
    % Ensure we have enough colors
    while size(colors, 1) < k_clusters
        colors = [colors; rand(1, 3)];
    end
    
    % Create colored segmentation visualization
    colored_segments = zeros(height, width, 3);
    for k = 1:k_clusters
        for channel = 1:3
            temp = colored_segments(:,:,channel);
            temp(cleaned_masks{k}) = colors(k, channel);
            colored_segments(:,:,channel) = temp;
        end
    end
    
    % Display results
    figure('Name', ['Image ' num2str(i) ' - Clustering Results']);
    
    % Original image
    subplot(2,2,1);
    imshow(img);
    title('Original Image');
    
    % Segmentation result (colored)
    subplot(2,2,2);
    imshow(colored_segments);
    title(['K-means Segmentation (k=' num2str(k_clusters) ')']);
    
    % Individual cluster visualization
    subplot(2,2,3);
    montage(cleaned_masks, 'Size', [1 k_clusters]);
    title('Individual Clusters (Cleaned)');
    
    % Overlay result
    subplot(2,2,4);
    imshow(img);
    hold on;
    h = imshow(colored_segments);
    set(h, 'AlphaData', 0.3);
    title('Segmentation Overlay');
    
    % Save results
    [~, filename, ext] = fileparts(imageFiles(i).name);
    
    % Save colored segmentation
    output_path_colored = fullfile(outputFolder, [filename '_segmented_colored' ext]);
    imwrite(colored_segments, output_path_colored);
    
    % Save individual cluster masks
    for k = 1:k_clusters
        output_path_cluster = fullfile(outputFolder, [filename '_cluster_' num2str(k) ext]);
        imwrite(cleaned_masks{k}, output_path_cluster);
    end
    
    % Create and save a blended result
    blended = img * 0.7 + uint8(colored_segments * 255) * 0.3;
    output_path_blended = fullfile(outputFolder, [filename '_segmented_blended' ext]);
    imwrite(uint8(blended), output_path_blended);
end

fprintf('Processing complete! Results saved in %s\n', outputFolder);
fprintf('Note: K-means clustering was performed in Lab color space for better color separation.\n');
fprintf('Small objects were removed to reduce noise in the segmentation.\n');
