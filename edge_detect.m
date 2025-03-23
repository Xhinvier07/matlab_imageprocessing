% Edge detection script to process images using various edge detection methods
% Input folder: images/processed_images
% Output folder: edge_detection

% Get input from user for number of images to process
num_images = input('Enter the number of images to process: ');

% Setup folders
inputFolder = 'images/processed_images';
outputFolder = 'edge_detection';

% Create output folder if it doesn't exist
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

% Get list of image files
imageFiles = dir(fullfile(inputFolder, '*.jpg')); % Adjust file extension if needed

% Validate user input
num_images = min(num_images, length(imageFiles));
fprintf('Processing %d images...\n', num_images);

% Process each image
for i = 1:num_images
    % Read image
    img = imread(fullfile(inputFolder, imageFiles(i).name));
    
    % Convert to grayscale if image is RGB
    if size(img, 3) == 3
        gray_img = rgb2gray(img);
    else
        gray_img = img;
    end
    
    % Apply different edge detection methods
    % Canny edge detection (best overall performance)
    edges_canny = edge(gray_img, 'Canny');
    
    % Sobel edge detection
    edges_sobel = edge(gray_img, 'Sobel');
    
    % Prewitt edge detection
    edges_prewitt = edge(gray_img, 'Prewitt');
    
    % Enhance edges using morphological operations
    % Create structuring elements
    se_dilate = strel('disk', 1);
    se_clean = strel('disk', 2);
    
    % Enhance Canny edges (our chosen best method)
    edges_enhanced = edges_canny;
    % Clean up small noise
    edges_enhanced = imclose(edges_enhanced, se_clean);
    % Dilate to make edges more prominent
    edges_enhanced = imdilate(edges_enhanced, se_dilate);
    
    % Display results
    figure('Name', ['Image ' num2str(i) ' - Edge Detection Comparison']);
    
    % Original image
    subplot(2,3,1);
    imshow(img);
    title('Original Image');
    
    % Grayscale image
    subplot(2,3,2);
    imshow(gray_img);
    title('Grayscale Image');
    
    % Canny edges
    subplot(2,3,3);
    imshow(edges_canny);
    title('Canny Edge Detection');
    
    % Sobel edges
    subplot(2,3,4);
    imshow(edges_sobel);
    title('Sobel Edge Detection');
    
    % Prewitt edges
    subplot(2,3,5);
    imshow(edges_prewitt);
    title('Prewitt Edge Detection');
    
    % Enhanced edges
    subplot(2,3,6);
    imshow(edges_enhanced);
    title('Enhanced Edges (Canny + Morphology)');
    
    % Save the enhanced edge detection result
    [~, filename, ext] = fileparts(imageFiles(i).name);
    
    % Save Canny edge detection result
    output_path_canny = fullfile(outputFolder, [filename '_edges_canny' ext]);
    imwrite(edges_canny, output_path_canny);
    
    % Save enhanced edge detection result
    output_path_enhanced = fullfile(outputFolder, [filename '_edges_enhanced' ext]);
    imwrite(edges_enhanced, output_path_enhanced);
    
    % Optional: Save other edge detection results
    output_path_sobel = fullfile(outputFolder, [filename '_edges_sobel' ext]);
    imwrite(edges_sobel, output_path_sobel);
    
    output_path_prewitt = fullfile(outputFolder, [filename '_edges_prewitt' ext]);
    imwrite(edges_prewitt, output_path_prewitt);
    
    % Create a combined visualization
    figure('Name', ['Image ' num2str(i) ' - Edge Detection Analysis']);
    
    % Original with enhanced edges overlay
    subplot(1,2,1);
    imshow(img);
    hold on;
    
    % Create edge overlay in red (fixed version)
    edge_overlay = zeros([size(edges_enhanced), 3]);
    edge_overlay(:,:,1) = edges_enhanced;  % Red channel
    h = imshow(edge_overlay);
    set(h, 'AlphaData', edges_enhanced * 0.5);
    title('Original with Edge Overlay');
    
    % Enhanced edges alone
    subplot(1,2,2);
    imshow(edges_enhanced);
    title('Enhanced Edge Detection');
end

fprintf('Processing complete! Results saved in %s\n', outputFolder);
fprintf('Note: Canny edge detection was chosen as the primary method due to its superior performance.\n');
fprintf('The enhanced edges use Canny detection with morphological operations for better results.\n');
