# 🖼️ Image Segmentation and Object Classification using MATLAB  

![MATLAB](https://img.shields.io/badge/MATLAB-Image%20Processing-blue?style=for-the-badge&logo=matlab)  
🚀 **An advanced image processing pipeline** that performs **color-based segmentation, object detection, and scene classification** using MATLAB.

---

## ✨ Project Overview  
This project explores **image processing techniques** to segment, detect, and classify objects in images. We implemented **color-based segmentation, edge detection, object tracking, and machine learning-based scene classification** using MATLAB.  

It includes **traditional image processing techniques** such as:
- **Color Segmentation** (RGB, HSV, YCbCr)  
- **Edge Detection** (Sobel, Canny, Prewitt)  
- **K-Means Clustering** for region-based segmentation  
- **Connected Component Analysis** for object detection  
- **Blob Detection** to identify objects and calculate centroids/bounding boxes  

Additionally, we implemented **machine learning & deep learning**:
- **Scene Classification** using Support Vector Machines (SVM) and K-Nearest Neighbors (KNN)  
- **CNN-based Image Classification** for identifying specific objects (e.g., ginger cats)  

---

## 🛠️ Implemented Features  
✅ **Color-Based Object Segmentation**  
   - Converts images to **HSV and YCbCr color spaces**  
   - Applies **adaptive color thresholding** to detect specific objects  
   - Uses **morphological operations** to refine segmented areas  

✅ **Edge Detection & Enhancement**  
   - Implements **Sobel, Canny, and Prewitt filters**  
   - Uses **morphological operations** (dilation, erosion) to enhance edges  

✅ **Object Detection & Blob Analysis**  
   - **Connected Component Analysis (CCA)** for labeling distinct objects  
   - Computes **centroids, bounding boxes, and area** for each detected object  

✅ **Scene Classification (KNN)**  
   - Extracts **color histograms & texture features** (GLCM)  
   - Trains a **machine learning model** to classify images 

✅ **CNN-Based Image Classification**  
   - Trains a **Convolutional Neural Network (CNN)** to detect specific objects

---

## 📂 How to Run the Project  
1️⃣ **Ensure MATLAB is Installed**  
   - Required Toolboxes:  
     - **Image Processing Toolbox**  
     - **Deep Learning Toolbox** (for CNN)  

2️⃣ **Placed Images in `images/`**  
   - Included with **5 images** with varied content (bird, cat, lego, cloud, strawberry, tree).  

3️⃣ **Run the Scripts**  
   - **Resize Image:** `resize.m`
   - **Color Segmentation:** `color_segmentation.m`  
   - **Object Detection:** `object_detection.m`  
   - **Scene Classification:** `scene_classification.m`  
   - **CNN-Based Classification:** `ginger_cat_classification.m` 
   - **....** 

---

## 📝 Authors  
👤 **Xhinvier07** → [GitHub](https://github.com/Xhinvier07)  
👤 **d-quint** → [GitHub](https://github.com/d-quint)  

🛠️ Developed as part of our **Finals Project in Image Processing**.  
