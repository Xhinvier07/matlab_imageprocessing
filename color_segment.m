% Color segmentation script to process images and highlight green objects
% Input folder: images/processed_images
% Output folder: color_segmentation

% Get input from user for number of images to process
num_images = input('Enter the number of images to process: ');

% Setup folders
inputFolder = 'images/processed_images';
outputFolder = 'color_segmentation';

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
    
    % Convert to different color spaces
    hsv_img = rgb2hsv(img);
    ycbcr_img = rgb2ycbcr(img);
    
    % Display original image and different color spaces
    figure('Name', ['Image ' num2str(i) ' - Color Spaces']);
    
    % Original RGB
    subplot(2,2,1); 
    imshow(img); 
    title('Original (RGB)');
    
    % HSV components
    subplot(2,2,2);
    imshow(hsv_img);
    title('HSV Color Space');
    
    % YCbCr components
    subplot(2,2,3);
    imshow(ycbcr_img);
    title('YCbCr Color Space');
    
    % Define green color threshold in HSV (Green Masking)
    % Hue values for green are approximately between 0.25 and 0.45
    % Adjust saturation and value thresholds as needed
    color_mask = (hsv_img(:,:,1) >= 0.167) & (hsv_img(:,:,1) <= 0.5) & ...  % Hue: 60° to 180°
             (hsv_img(:,:,2) >= 0.2) & ...  % Allow lower saturation for muted greens
             (hsv_img(:,:,3) >= 0.15);      % Include darker greens
    
    % Yellow Mask
    % Define yellow color threshold in HSV (capturing all yellow shades)
    % color_mask = (hsv_img(:,:,1) >= 0.111) & (hsv_img(:,:,1) <= 0.222) & ...  % Hue: 40° to 80°
    % (hsv_img(:,:,2) >= 0.2) & ...  % Allow pastel to vibrant yellows
    % (hsv_img(:,:,3) >= 0.2);       % Include dark and bright yellows

    % Red Mask 
    % Define red color threshold in HSV (covering all red shades)
    % red_mask = ((hsv_img(:,:,1) >= 0.958) | (hsv_img(:,:,1) <= 0.042)) & ... % Hue: 345°-360° OR 0°-15°
    %       (hsv_img(:,:,2) >= 0.2) & ...  % Allow low-saturation reds (muted, dark reds)
    %       (hsv_img(:,:,3) >= 0.2);       % Include both dark and bright reds


    % Apply morphological operations to clean up the mask
    se = strel('disk', 3);
    color_mask = imopen(color_mask, se);
    color_mask = imclose(color_mask, se);
    
    % Convert original image to grayscale
    gray_img = rgb2gray(img);
    
    % Create output image (initially grayscale)
    output_img = repmat(gray_img, [1 1 3]);
    
    % Keep original colors only for green regions (fixed indexing)
    for channel = 1:3
        temp = output_img(:,:,channel);
        channel_data = img(:,:,channel);
        temp(color_mask) = channel_data(color_mask);
        output_img(:,:,channel) = temp;
    end
    
    % Display segmentation result
    subplot(2,2,4);
    imshow(output_img);
    title('Green Segmentation Result');
    
    % Save the processed image
    [~, filename, ext] = fileparts(imageFiles(i).name);
    output_path = fullfile(outputFolder, [filename '_segmented' ext]);
    imwrite(output_img, output_path);
    
    % Display individual channels of each color space in a new figure
    figure('Name', ['Image ' num2str(i) ' - Color Channel Components']);
    
    % RGB Channels
    subplot(3,3,1); imshow(img(:,:,1)); title('R Channel');
    subplot(3,3,2); imshow(img(:,:,2)); title('G Channel');
    subplot(3,3,3); imshow(img(:,:,3)); title('B Channel');
    
    % HSV Channels
    subplot(3,3,4); imshow(hsv_img(:,:,1)); title('Hue');
    subplot(3,3,5); imshow(hsv_img(:,:,2)); title('Saturation');
    subplot(3,3,6); imshow(hsv_img(:,:,3)); title('Value');
    
    % YCbCr Channels
    subplot(3,3,7); imshow(ycbcr_img(:,:,1)); title('Y (Luminance)');
    subplot(3,3,8); imshow(ycbcr_img(:,:,2)); title('Cb (Blue-Yellow)');
    subplot(3,3,9); imshow(ycbcr_img(:,:,3)); title('Cr (Red-Green)');
end

fprintf('Processing complete! Results saved in %s\n', outputFolder);
