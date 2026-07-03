# Project-1-CNN: CIFAR-10 Image Classification

## 1. Title
Project-1-CNN: Iterative CNN Modeling on CIFAR-10

## 2. About
This project explores how a Convolutional Neural Network (CNN) can be improved step by step for image classification on CIFAR-10. Starting from a base model, five model variants are tested to measure the impact of architecture choices (Batch Normalization, Dropout, number of convolutional blocks, augmentation, and optimizer changes).

The work is documented across multiple notebooks, where each notebook represents one iteration of the model development process.

## 3. Problem Statement
- Dataset used: CIFAR-10
- Task: Multi-class image classification (10 classes)
- Goal: Build and compare CNN variants to maximize validation/test performance while reducing overfitting

In short: given a 32x32 RGB image, predict its correct class label.

## 4. Dataset
- Name: CIFAR-10
- Source: Canadian Institute for Advanced Research (hosted via TensorFlow/Keras)
- Size: 60,000 color images (32x32), 10 classes
- Class labels: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- Project split: 70% train, 15% validation, 15% test (stratified)
- Dataset link: https://www.cs.toronto.edu/~kriz/cifar.html
- Keras loader reference: https://www.tensorflow.org/api_docs/python/tf/keras/datasets/cifar10/load_data
- License note: CIFAR-10 is publicly available for research/educational use; see the dataset source page for terms and citation details

## 5. Model Architecture

### Base architecture idea
- Sequential CNN with stacked Conv2D + MaxPooling blocks
- Fully connected head ending in Softmax(10)
- Trained with categorical crossentropy

### Iterative architecture changes
- Base model: 4 convolutional blocks (32, 64, 128, 256), no regularization
- Model 1: add Batch Normalization layers
- Model 2: switch to 3 blocks + Dropout
- Model 3: combine Batch Normalization + Dropout + data augmentation
- Model 4: transfer-learning comparison notebook (reuses Model 3 CNN, adds VGG16 experiment)
- Model 5: 4 blocks with Batch Normalization + Dropout, SGD optimizer, no augmentation

### CNN flow diagram (representative)
```mermaid
flowchart LR
		A[Input 32x32x3] --> B[Conv Block 1]
		B --> C[Conv Block 2]
		C --> D[Conv Block 3]
		D --> E[Optional Conv Block 4]
		E --> F[Flatten]
		F --> G[Dense 512]
		G --> H[Dropout]
		H --> I[Dense 10 Softmax]
```

## 6. Results

### Main metrics (best epoch by validation loss)

| Model | Validation Accuracy | Validation Loss | Test Accuracy |
|---|---:|---:|---:|
| Base | 0.7163 | 0.8296 | 0.7048 |
| Model 1 | 0.7789 | 0.6821 | 0.7728 |
| Model 2 | 0.7884 | 0.6211 | 0.7869 |
| Model 3 | 0.8249 | 0.5037 | 0.8249 |
| Model 4 (same CNN as Model 3; transfer-learning comparison notebook) | 0.8249 | 0.5037 | 0.8249 |
| Model 5 | 0.7860 | 0.6213 | 0.7788 |

Best custom CNN: Model 3 with 82.49% test accuracy.

### Transfer learning comparison (Model 4 notebook)
- Note: Model 4 does not introduce a new custom CNN architecture. It reuses the Model 3 CNN and adds a separate VGG16 comparison experiment.
- VGG16 test accuracy: 0.6124
- VGG16 validation accuracy at best epoch: 0.6201
- VGG16 validation loss at best epoch: 1.1035

### Confusion matrix and sample outputs
- Confusion matrices are generated in each model notebook using sklearn metrics + seaborn heatmaps.
- Sample output visualizations include:
	- sample CIFAR-10 training images
	- class distribution plots
	- training/validation accuracy-loss curves
	- confusion matrix heatmaps

## 7. Setup and Installation

### Clone repository
```bash
git clone <your-repo-url>
cd Project-1-CNN
```

### Create virtual environment
```bash
python -m venv .venv
```

Windows (PowerShell):
```bash
.venv\Scripts\Activate.ps1
```

macOS/Linux:
```bash
source .venv/bin/activate
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run notebooks
1. Open notebooks locally in Jupyter or in Google Colab.
2. Run cells from top to bottom.
3. CIFAR-10 is loaded directly with tensorflow.keras.datasets.cifar10.

## Repository Structure

- Base_Model_Project_1_CNN.ipynb
- Model_1 Project_1_CNN.ipynb
- Model_2 Project_1_CNN.ipynb
- Model_3Project_1_CNN.ipynb
- Model_4Project_1_CNN.ipynb
- Model_5Project_1_CNN.ipynb
- Project-1-CNN.ipynb
