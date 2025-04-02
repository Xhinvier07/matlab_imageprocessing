# Image Processing Project

This repository contains various image processing scripts and tools for tasks such as noise reduction, segmentation, classification, object detection, and more. The project is implemented in MATLAB and is organized into different modules for specific functionalities.

## Table of Contents

- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Usage](#usage)
  - [Preprocessing](#preprocessing)
  - [Segmentation](#segmentation)
  - [Classification](#classification)
  - [Object Detection](#object-detection)
  - [Noise Reduction](#noise-reduction)
  - [Edge Detection](#edge-detection)
- [Scripts Overview](#scripts-overview)
- [License](#license)

---

## Project Structure

```
image_processing/
├── ginger_cat_classification/
│   ├── untitled.fig
│   ├── ginger_cat_detector.mat
├── scene_classification/
│   ├── knn_model.mat
├── noise_reduction/
│   ├── cat_metrics.txt
│   ├── bird_metrics.txt
│   ├── block_metrics.txt
│   ├── rainy_metrics.txt
│   ├── tree_metrics.txt
│   ├── strawberry_metrics.txt
├── object_detection/
│   ├── object_details_bird.txt
│   ├── object_details_block.txt
│   ├── object_details_cat.txt
│   ├── object_details_tree.txt
│   ├── object_details_strawberry.txt
├── cluster_segment.m
├── cnn_classify.m
├── cnn_train.m
├── detectGingerCat.m
├── edge_detect.m
├── noise_reduction.m
├── object_detect.m
├── resize.m
├── scene_classify_predict.m
├── scene_classify_train.m
├── color_segment.m
├── LICENSE
```

---

## Requirements

- MATLAB (with Image Processing Toolbox and Deep Learning Toolbox)
- A dataset of images for training and testing
- Pre-trained models (e.g., `ginger_cat_detector.mat`, `knn_model.mat`) if not training from scratch

---

## Usage

### Preprocessing

1. **Resize Images**: Use `resize.m` to resize images to a fixed size (e.g., 512x512) for consistent processing.
   ```matlab
   run('resize.m');
   ```

### Segmentation

2. **Color-Based Segmentation**: Use `color_segment.m` to segment images based on specific color thresholds (e.g., green, red, yellow).
   ```matlab
   run('color_segment.m');
   ```

3. **Clustering-Based Segmentation**: Use `cluster_segment.m` to segment images using K-means clustering.
   ```matlab
   run('cluster_segment.m');
   ```

### Classification

4. **Train Scene Classifier**: Use `scene_classify_train.m` to train a KNN classifier for scene classification.
   ```matlab
   run('scene_classify_train.m');
   ```

5. **Predict Scene Class**: Use `scene_classify_predict.m` to classify a new image using the trained KNN model.
   ```matlab
   run('scene_classify_predict.m');
   ```

6. **Ginger Cat Classification**:
   - Train the CNN model using `cnn_train.m`.
   - Classify images using `cnn_classify.m`.

   ```matlab
   run('cnn_train.m');
   run('cnn_classify.m');
   ```

### Object Detection

7. **Object Detection**: Use `object_detect.m` to detect objects in images based on color thresholds and bounding box analysis.
   ```matlab
   run('object_detect.m');
   ```

### Noise Reduction

8. **Noise Reduction**: Use `noise_reduction.m` to apply various noise reduction techniques (e.g., median, Gaussian, bilateral filtering).
   ```matlab
   run('noise_reduction.m');
   ```

### Edge Detection

9. **Edge Detection**: Use `edge_detect.m` to detect edges in images using methods like Canny, Sobel, and Prewitt.
   ```matlab
   run('edge_detect.m');
   ```

---

## Scripts Overview

### Key Scripts

- **`resize.m`**: Resizes images to a fixed size for consistent processing.
- **`color_segment.m`**: Segments images based on color thresholds.
- **`cluster_segment.m`**: Performs clustering-based segmentation using K-means.
- **`scene_classify_train.m`**: Trains a KNN classifier for scene classification.
- **`scene_classify_predict.m`**: Predicts the class of a new image using the trained KNN model.
- **`cnn_train.m`**: Trains a CNN model for ginger cat classification.
- **`cnn_classify.m`**: Classifies images as "Ginger Cat" or "Not Ginger Cat" using the trained CNN model.
- **`object_detect.m`**: Detects objects in images using color thresholds and bounding box analysis.
- **`noise_reduction.m`**: Applies noise reduction techniques to images.
- **`edge_detect.m`**: Detects edges in images using various edge detection methods.

### Supporting Files

- **Pre-trained Models**:
  - `ginger_cat_classification/ginger_cat_detector.mat`: Pre-trained CNN model for ginger cat classification.
  - `scene_classification/knn_model.mat`: Pre-trained KNN model for scene classification.

- **Metrics and Results**:
  - `noise_reduction/*.txt`: Metrics for noise reduction techniques.
  - `object_detection/*.txt`: Object detection results for specific images.

---

## License

This project is licensed under the Creative Commons CC0 1.0 Universal license. See the [LICENSE](LICENSE) file for details.