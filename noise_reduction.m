% Noise reduction and image enhancement script
% Input folder: images/processed_images
% Output folder: noise_reduction

% Get input from user for number of images to process
num_images = input('Enter the number of images to process: ');

% Setup folders
inputFolder = 'images/processed_images';
outputFolder = 'noise_reduction';

% Create output folder if it doesn't exist
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

% Get list of image files
imageFiles = dir(fullfile(inputFolder, '*.jpg')); % Adjust file extension if needed

% Validate user input
num_images = min(num_images, length(imageFiles));
fprintf('Processing %d images...\n', num_images);

% Function to apply median filtering
function filtered_img = applyMedianFilter(img, window_size)
    if size(img, 3) == 3
        % Apply median filter to each color channel
        filtered_img = zeros(size(img), 'uint8');
        for channel = 1:3
            filtered_img(:,:,channel) = medfilt2(img(:,:,channel), [window_size window_size]);
        end
    else
        filtered_img = medfilt2(img, [window_size window_size]);
    end
end

% Function to apply Gaussian filtering
function filtered_img = applyGaussianFilter(img, sigma)
    if size(img, 3) == 3
        % Apply Gaussian filter to each color channel
        filtered_img = imgaussfilt(img, sigma);
    else
        filtered_img = imgaussfilt(img, sigma);
    end
end

% Function to apply bilateral filtering
function filtered_img = applyBilateralFilter(img, sigma_spatial, sigma_range)
    if size(img, 3) == 3
        % Apply bilateral filter to each color channel
        filtered_img = zeros(size(img), 'uint8');
        for channel = 1:3
            filtered_img(:,:,channel) = imbilatfilt(img(:,:,channel), ...
                sigma_range, sigma_spatial);
        end
    else
        filtered_img = imbilatfilt(img, sigma_range, sigma_spatial);
    end
end

% Function to enhance contrast using CLAHE
function enhanced_img = enhanceContrast(img)
    if size(img, 3) == 3
        % Convert to LAB color space
        lab_img = rgb2lab(img);
        
        % Apply CLAHE to L channel
        L = lab_img(:,:,1);
        L = rescale(L, 0, 1);
        L = adapthisteq(L, 'ClipLimit', 0.02, 'Distribution', 'rayleigh');
        
        % Convert back to RGB
        lab_img(:,:,1) = L * 100;
        enhanced_img = lab2rgb(lab_img);
        enhanced_img = im2uint8(enhanced_img);
    else
        enhanced_img = adapthisteq(img, 'ClipLimit', 0.02, 'Distribution', 'rayleigh');
    end
end

% Function to adjust image sharpness
function sharpened_img = adjustSharpness(img, amount)
    if size(img, 3) == 3
        % Convert to LAB color space
        lab_img = rgb2lab(img);
        
        % Sharpen L channel
        L = lab_img(:,:,1);
        L = imsharpen(L, 'Amount', amount, 'Radius', 1, 'Threshold', 0.05);
        
        % Convert back to RGB
        lab_img(:,:,1) = L;
        sharpened_img = lab2rgb(lab_img);
        sharpened_img = im2uint8(sharpened_img);
    else
        sharpened_img = imsharpen(img, 'Amount', amount, 'Radius', 1, 'Threshold', 0.05);
    end
end

% Process each image
for i = 1:num_images
    % Read image
    img = imread(fullfile(inputFolder, imageFiles(i).name));
    
    % Convert to double for processing
    img_double = im2double(img);
    
    % Apply noise reduction techniques
    % 1. Median filtering (good for salt-and-pepper noise)
    median_filtered = applyMedianFilter(img, 3);
    
    % 2. Gaussian filtering (good for Gaussian noise)
    gaussian_filtered = applyGaussianFilter(img, 1.0);
    
    % 3. Bilateral filtering (edge-preserving smoothing)
    bilateral_filtered = applyBilateralFilter(img, 3, 0.1);
    
    % Image enhancement
    % 1. Contrast enhancement using CLAHE
    contrast_enhanced = enhanceContrast(img);
    
    % 2. Sharpness adjustment
    sharpened = adjustSharpness(img, 1.5);
    
    % Create visualization
    figure('Name', ['Image ' num2str(i) ' - Noise Reduction & Enhancement']);
    
    % Original image
    subplot(2,3,1);
    imshow(img);
    title('Original Image');
    
    % Median filtered
    subplot(2,3,2);
    imshow(median_filtered);
    title('Median Filtered');
    
    % Gaussian filtered
    subplot(2,3,3);
    imshow(gaussian_filtered);
    title('Gaussian Filtered');
    
    % Bilateral filtered
    subplot(2,3,4);
    imshow(bilateral_filtered);
    title('Bilateral Filtered');
    
    % Contrast enhanced
    subplot(2,3,5);
    imshow(contrast_enhanced);
    title('Contrast Enhanced');
    
    % Sharpened
    subplot(2,3,6);
    imshow(sharpened);
    title('Sharpened');
    
    % Save results
    [~, filename, ext] = fileparts(imageFiles(i).name);
    
    % Save all processed versions
    imwrite(median_filtered, fullfile(outputFolder, [filename '_median' ext]));
    imwrite(gaussian_filtered, fullfile(outputFolder, [filename '_gaussian' ext]));
    imwrite(bilateral_filtered, fullfile(outputFolder, [filename '_bilateral' ext]));
    imwrite(contrast_enhanced, fullfile(outputFolder, [filename '_contrast' ext]));
    imwrite(sharpened, fullfile(outputFolder, [filename '_sharp' ext]));
    
    % Save visualization
    output_path_vis = fullfile(outputFolder, [filename '_comparison' ext]);
    frame = getframe(gcf);
    imwrite(frame.cdata, output_path_vis);
    
    % Calculate and save quality metrics
    output_path_metrics = fullfile(outputFolder, [filename '_metrics.txt']);
    fid = fopen(output_path_metrics, 'w');
    fprintf(fid, 'Image Processing Metrics for %s\n\n', imageFiles(i).name);
    
    % Calculate PSNR for each method
    fprintf(fid, 'Peak Signal-to-Noise Ratio (PSNR):\n');
    fprintf(fid, '--------------------------------\n');
    fprintf(fid, 'Median Filter: %.2f dB\n', psnr(median_filtered, img));
    fprintf(fid, 'Gaussian Filter: %.2f dB\n', psnr(gaussian_filtered, img));
    fprintf(fid, 'Bilateral Filter: %.2f dB\n', psnr(bilateral_filtered, img));
    fprintf(fid, 'Contrast Enhanced: %.2f dB\n', psnr(contrast_enhanced, img));
    fprintf(fid, 'Sharpened: %.2f dB\n\n', psnr(sharpened, img));
    
    % Calculate MSE for each method
    fprintf(fid, 'Mean Squared Error (MSE):\n');
    fprintf(fid, '------------------------\n');
    fprintf(fid, 'Median Filter: %.2f\n', immse(median_filtered, img));
    fprintf(fid, 'Gaussian Filter: %.2f\n', immse(gaussian_filtered, img));
    fprintf(fid, 'Bilateral Filter: %.2f\n', immse(bilateral_filtered, img));
    fprintf(fid, 'Contrast Enhanced: %.2f\n', immse(contrast_enhanced, img));
    fprintf(fid, 'Sharpened: %.2f\n', immse(sharpened, img));
    
    fclose(fid);
end

fprintf('Processing complete! Results saved in %s\n', outputFolder);
fprintf('Applied techniques:\n');
fprintf('1. Median Filtering (removes salt-and-pepper noise)\n');
fprintf('2. Gaussian Filtering (removes Gaussian noise)\n');
fprintf('3. Bilateral Filtering (edge-preserving smoothing)\n');
fprintf('4. Contrast Enhancement (CLAHE)\n');
fprintf('5. Sharpness Adjustment\n');
