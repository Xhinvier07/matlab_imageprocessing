% Load and resize images
imageFolder = 'images/';
processedFolder = fullfile(imageFolder, 'processed_images');

% Create the processed images folder if it doesn't exist
if ~exist(processedFolder, 'dir')
    mkdir(processedFolder);
end

imageFiles = dir(fullfile(imageFolder, '*.jpg')); % or other format
targetSize = [512, 512];

for i = 1:length(imageFiles)
    % Read image
    img = imread(fullfile(imageFolder, imageFiles(i).name));
    
    % Resize image
    img_resized = imresize(img, targetSize);
    
    % Save the resized image to the new folder
    imwrite(img_resized, fullfile(processedFolder, imageFiles(i).name));
    
    % Display the resized image
    figure; imshow(img_resized); title(['Resized Image ' num2str(i)]);
end