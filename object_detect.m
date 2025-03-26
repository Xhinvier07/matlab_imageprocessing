% Object Detection Script with Blob Analysis, External Thresholds, and NMS for Cat

% Setup folders
inputFolder = 'images/processed_images';
outputFolder = 'object_detection';

% Create output folder if it doesn't exist
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

% Get list of image files
imageFiles = dir(fullfile(inputFolder, '*.jpg')); % Adjust file extension if needed

% Process each image
for i = 1:length(imageFiles)
    % Read the image
    fullImagePath = fullfile(inputFolder, imageFiles(i).name);
    RGB = imread(fullImagePath);
    
    % Extract filename (without extension)
    [~, filename, ~] = fileparts(imageFiles(i).name);
    
    % Determine color space and thresholds based on filename
    switch lower(filename)
        case 'bird'
            % Convert RGB image to HSV color space
            I = rgb2hsv(RGB);
            channel1Min = 0.334;
            channel1Max = 0.668;
            channel2Min = 0.310;
            channel2Max = 1.000;
            channel3Min = 0.000;
            channel3Max = 1.000;
            
            % Standard AND condition
            sliderBW = (I(:,:,1) >= channel1Min ) & (I(:,:,1) <= channel1Max) & ...
                       (I(:,:,2) >= channel2Min ) & (I(:,:,2) <= channel2Max) & ...
                       (I(:,:,3) >= channel3Min ) & (I(:,:,3) <= channel3Max);
        
        case 'block'
            % Convert RGB image to HSV color space
            I = rgb2hsv(RGB);
            channel1Min = 0.047;
            channel1Max = 0.039;
            channel2Min = 0.116;
            channel2Max = 1.000;
            channel3Min = 0.000;
            channel3Max = 1.000;
            
            % Special OR condition for first channel
            sliderBW = ( (I(:,:,1) >= channel1Min) | (I(:,:,1) <= channel1Max) ) & ...
                       (I(:,:,2) >= channel2Min ) & (I(:,:,2) <= channel2Max) & ...
                       (I(:,:,3) >= channel3Min ) & (I(:,:,3) <= channel3Max);
        
        case 'cat'
            % Convert RGB image to HSV color space
            I = rgb2hsv(RGB);
            channel1Min = 0.000;
            channel1Max = 1.000;
            channel2Min = 0.000;
            channel2Max = 1.000;
            channel3Min = 0.843;
            channel3Max = 1.000;
            
            % Standard AND condition
            sliderBW = (I(:,:,1) >= channel1Min ) & (I(:,:,1) <= channel1Max) & ...
                       (I(:,:,2) >= channel2Min ) & (I(:,:,2) <= channel2Max) & ...
                       (I(:,:,3) >= channel3Min ) & (I(:,:,3) <= channel3Max);
        
        case 'strawberry'
            % Use RGB color space directly
            I = RGB;
            channel1Min = 122.000;
            channel1Max = 255.000;
            channel2Min = 0.000;
            channel2Max = 255.000;
            channel3Min = 0.000;
            channel3Max = 255.000;
            
            % Standard AND condition
            sliderBW = (I(:,:,1) >= channel1Min ) & (I(:,:,1) <= channel1Max) & ...
                       (I(:,:,2) >= channel2Min ) & (I(:,:,2) <= channel2Max) & ...
                       (I(:,:,3) >= channel3Min ) & (I(:,:,3) <= channel3Max);
        
        case 'tree'
            % Use RGB color space directly
            I = RGB;
            channel1Min = 4.000;
            channel1Max = 255.000;
            channel2Min = 9.000;
            channel2Max = 101.000;
            channel3Min = 0.000;
            channel3Max = 255.000;
            
            % Standard AND condition
            sliderBW = (I(:,:,1) >= channel1Min ) & (I(:,:,1) <= channel1Max) & ...
                       (I(:,:,2) >= channel2Min ) & (I(:,:,2) <= channel2Max) & ...
                       (I(:,:,3) >= channel3Min ) & (I(:,:,3) <= channel3Max);
        
        otherwise
            % Skip or handle undefined cases
            fprintf('No threshold defined for image: %s\n', filename);
            continue;
    end
    
    % Convert to binary
    BW = sliderBW;
    
    % Perform morphological operations to clean up the binary image
    % Remove small noise and fill holes
    BW = bwareaopen(BW, 700);  % Remove small objects smaller than 700 pixels
    BW = imfill(BW, 'holes');
    
    % Perform connected component analysis
    [labeledImage, numObjects] = bwlabel(BW);
    
    % Measure properties of detected objects
    stats = regionprops(labeledImage, 'Centroid', 'Area', 'BoundingBox');
    
    % Prepare text file for object details
    detailsFileName = fullfile(outputFolder, ['object_details_' filename '.txt']);
    detailsFid = fopen(detailsFileName, 'w');
    fprintf(detailsFid, 'Object Detection Results for %s\n', filename);
    fprintf(detailsFid, '======================================\n');
    fprintf(detailsFid, 'Color Thresholds:\n');
    fprintf(detailsFid, '  Channel 1 Min: %.3f, Max: %.3f\n', channel1Min, channel1Max);
    fprintf(detailsFid, '  Channel 2 Min: %.3f, Max: %.3f\n', channel2Min, channel2Max);
    fprintf(detailsFid, '  Channel 3 Min: %.3f, Max: %.3f\n', channel3Min, channel3Max);
    fprintf(detailsFid, '======================================\n');
    
    % Create a figure that won't close automatically
    figure('Name', ['Object Detection - ' filename], 'NumberTitle', 'off', 'Units', 'normalized', 'Position', [0.1 0.1 0.8 0.8]);
    
    % Display original image
    imshow(RGB);
    title('Object Detection Results');
    hold on;
    
    % Apply NMS for cat detection
    if strcmpi(filename, 'cat') && numObjects > 1
        % Extract bounding boxes and create scores (using area as score)
        boxes = zeros(numObjects, 4);
        scores = zeros(numObjects, 1);
        
        for j = 1:numObjects
            boxes(j,:) = stats(j).BoundingBox;
            scores(j) = stats(j).Area;
        end
        
        % Convert bounding boxes from [x, y, width, height] to [x1, y1, x2, y2]
        boxes_xyxy = [boxes(:,1), boxes(:,2), boxes(:,1)+boxes(:,3), boxes(:,2)+boxes(:,4)];
        
        % Define a function to calculate IoU
        calculateIoU = @(box1, box2) rectint(box1, box2) / ...
            (box1(3)*box1(4) + box2(3)*box2(4) - rectint(box1, box2));
        
        % Define overlap threshold for NMS
        overlapThreshold = 0.3;
        
        % Initialize index vector for NMS
        keep = true(numObjects, 1);
        
        % Sort boxes by score (area)
        [~, sortedIndices] = sort(scores, 'descend');
        
        % NMS loop
        for j = 1:numObjects-1
            if keep(sortedIndices(j))
                box1 = boxes(sortedIndices(j), :);
                
                for k = j+1:numObjects
                    if keep(sortedIndices(k))
                        box2 = boxes(sortedIndices(k), :);
                        
                        % Calculate IoU
                        x1 = max(box1(1), box2(1));
                        y1 = max(box1(2), box2(2));
                        x2 = min(box1(1)+box1(3), box2(1)+box2(3));
                        y2 = min(box1(2)+box1(4), box2(2)+box2(4));
                        
                        % Check for overlap
                        if x2 > x1 && y2 > y1
                            intersection = (x2-x1) * (y2-y1);
                            area1 = box1(3) * box1(4);
                            area2 = box2(3) * box2(4);
                            union = area1 + area2 - intersection;
                            iou = intersection / union;
                            
                            if iou > overlapThreshold
                                keep(sortedIndices(k)) = false;
                            end
                        end
                    end
                end
            end
        end
        
        % Create merged bounding cat image
        % Find the extreme coordinates to create a single bounding box that includes all detections
        minX = min(boxes(:,1));
        minY = min(boxes(:,2));
        maxX = max(boxes(:,1) + boxes(:,3));
        maxY = max(boxes(:,2) + boxes(:,4));
        
        % Create a single bounding box that encompasses all detections
        catBox = [minX, minY, maxX - minX, maxY - minY];
        
        % Print merged cat detection
        fprintf('Merged cat detection: [x y width height] = [%.2f %.2f %.2f %.2f]\n', ...
            catBox(1), catBox(2), catBox(3), catBox(4));
        
        % Draw the merged bounding box
        rectangle('Position', catBox, 'EdgeColor', 'r', 'LineWidth', 3);
        
        % Find center of the merged box for centroid
        catCentroid = [catBox(1) + catBox(3)/2, catBox(2) + catBox(4)/2];
        plot(catCentroid(1), catCentroid(2), 'b+', 'MarkerSize', 12);
        
        % Write details to text file
        fprintf(detailsFid, '\nMerged Cat Detection:\n');
        fprintf(detailsFid, '  Centroid: (%.2f, %.2f)\n', catCentroid(1), catCentroid(2));
        fprintf(detailsFid, '  Bounding Box: [x y width height] = [%.2f %.2f %.2f %.2f]\n', ...
            catBox(1), catBox(2), catBox(3), catBox(4));
        fprintf(detailsFid, '  Area: %.2f pixels\n', catBox(3) * catBox(4));
        
        % Also draw original detections with thinner lines for comparison
        for j = 1:numObjects
            % Draw bounding box with dashed line to show original detections
            rectangle('Position', stats(j).BoundingBox, 'EdgeColor', 'r', 'LineWidth', 1, 'LineStyle', ':');
            % Draw centroid
            plot(stats(j).Centroid(1), stats(j).Centroid(2), 'b+', 'MarkerSize', 6);
        end
    else
        % For non-cat images
        % Print object detection results
        fprintf('Image %s: Detected %d objects\n', filename, numObjects);
        
        % Annotate objects on the image
        for j = 1:numObjects
            % Draw bounding box on the original image
            rectangle('Position', stats(j).BoundingBox, 'EdgeColor', 'r', 'LineWidth', 2);
            
            % Draw centroid
            plot(stats(j).Centroid(1), stats(j).Centroid(2), 'b+', 'MarkerSize', 10);
            
            % Print and save object details
            fprintf('Object %d:\n', j);
            fprintf('  Centroid: (%.2f, %.2f)\n', stats(j).Centroid(1), stats(j).Centroid(2));
            fprintf('  Area: %.2f pixels\n', stats(j).Area);
            
            % Write details to text file
            fprintf(detailsFid, '\nObject %d:\n', j);
            fprintf(detailsFid, '  Centroid: (%.2f, %.2f)\n', stats(j).Centroid(1), stats(j).Centroid(2));
            fprintf(detailsFid, '  Area: %.2f pixels\n', stats(j).Area);
            fprintf(detailsFid, '  Bounding Box: [x y width height] = [%.2f %.2f %.2f %.2f]\n', ...
                stats(j).BoundingBox(1), stats(j).BoundingBox(2), stats(j).BoundingBox(3), stats(j).BoundingBox(4));
        end
    end
    
    % Close text file
    fclose(detailsFid);
    
    % Save the annotated image
    outputImagePath = fullfile(outputFolder, ['annotated_' filename '.jpg']);
    saveas(gcf, outputImagePath);
    
    % Pause to keep the figure open (press any key to continue to next image)
    fprintf('Press any key to continue to the next image...\n');
    waitforbuttonpress;
    close(gcf);
end

fprintf('Object detection complete. Annotated images and details saved in %s\n', outputFolder);