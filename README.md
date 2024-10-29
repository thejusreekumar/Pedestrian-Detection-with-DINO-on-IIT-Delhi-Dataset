# Pedestrian Detection with DINO on IIT Delhi Dataset

This repository contains code and instructions for training and evaluating the DINO object detection model on a pedestrian dataset collected within the IIT Delhi campus. The dataset consists of 200 images annotated in COCO format.

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset Preparation](#dataset-preparation)
3. [Environment Setup](#environment-setup)
4. [Pre-trained Model Download](#pre-trained-model-download)
5. [Training and Fine-tuning](#training-and-fine-tuning)
6. [Evaluation and Results](#evaluation-and-results)
7. [Visualization](#visualization)
8. [Report](#report)
9. [References](#references)

---

### 1. Introduction

This project aims to train and fine-tune the DINO object detection model for detecting pedestrians in images. We evaluate the model using bounding box Average Precision (AP) values and analyze the results, including error cases and attention maps.

### 2. Dataset Preparation

- **Data Source**: The dataset is provided as a link and should be downloaded into a local or cloud environment (e.g., Google Drive).
- **Splitting**: Split the dataset into 160 images for training and 40 for validation. You can use the following code snippet to automate the process:

   ```python
   # Import necessary libraries
   from sklearn.model_selection import train_test_split
   import shutil
   import os

   # Load images and annotations, then split
   image_files = [img['file_name'] for img in annotations['images']]
   train_files, val_files = train_test_split(image_files, test_size=0.2, random_state=42)

   # Move files to train and validation folders
   for file in train_files:
       shutil.move(os.path.join(dataset_path, file), train_dir)
   for file in val_files:
       shutil.move(os.path.join(dataset_path, file), val_dir)
### 3.Environment Setup
- **Clone the Repository**:
  git clone https://github.com/IDEA-Research/DINO.git
  cd DINO
-**Install dependencies**:
  pip install -r requirements.txt
-**Configure Paths**:
  Update the configs/dino_r50.yaml file to specify the paths to your dataset (both training and validation) and pre-trained model weights.

### 4. Pre-trained Model Download
Download the DINO model with ResNet-50 backbone from the DINO repository and save it in the designated directory. Update configs/dino_r50.yaml with the path to this pre-trained model.

### 5. Training and Fine-tuning
Initial Evaluation: Run the pre-trained model on the validation set:

bash
python tools/test_net.py --config-file configs/dino_r50.yaml --eval-only
Fine-tune the Model: Fine-tune the model on the training dataset to improve pedestrian detection accuracy.

bash
python tools/train_net.py --config-file configs/dino_r50.yaml
Re-evaluate: After fine-tuning, run another evaluation on the validation set:

bash
python tools/test_net.py --config-file configs/dino_r50.yaml --eval-only

### 6. Evaluation and Results
Bounding Box Average Precision (AP): Calculate the AP scores for pedestrian detection using the COCO evaluation metrics. Record the AP values for the validation set before and after fine-tuning.
Error Analysis: Identify cases where the model struggled to detect pedestrians and analyze common errors (e.g., false positives and missed detections).

### 7. Visualization
Bounding Box Visualizations: Display bounding boxes on validation images to assess the detection quality.
Attention Maps: Visualize attention maps to understand the model’s focus areas when detecting pedestrians.

### 8. Report
Include a report.pdf in this repository detailing:

Experimental Setup: Dataset, model configurations, training process, and evaluation metrics.
Results: AP scores for both pre-trained and fine-tuned models.
Visualizations: Bounding box predictions, attention maps, and loss graphs.
Error Analysis: Observations on common failure cases and potential improvements.

### 9. References
DINO Repository
COCO Evaluation Metrics
