# Rock Classification with a RAMAN Spectrometer

- **Author:** Tanmay Talreja (m12519565)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![Status](https://img.shields.io/badge/Status-Completed-green)

This project implements robust machine learning pipelines for classifying minerals effectively using both **image data** and **spectral profile data** at varying frame rates (1fps and 30fps). It benchmarks various architectures including **Random Forest**, **ResNet50**, **MobileViT**, **1D CNNs**, and **MLPs** to determine the optimal approach for mineral identification.

## Table of Contents
- [Directory Structure](#directory-structure)
- [Prerequisites](#prerequisites)
- [Setup and Installation](#setup-and-installation)
- [How to Run](#how-to-run)
- [Model Architectures](#model-architectures)
- [Results](#results)

## Directory Structure

```
├── NNATT dataset/       # Source dataset containing mineral images and profile CSVs
├── docs/                # Report and Presentation
├── models/              # Saved trained models (.pth, .joblib)
├── results/             # Generated visualizations (confusion matrices, accuracy plots)
├── scripts/             # Training notebooks
│   ├── images 1fps.ipynb
│   ├── images 30fps.ipynb
│   ├── profiles 1fps.ipynb
│   └── profiles 30fps.ipynb
├── test/                # Testing notebooks
│   ├── test_images_1fps.ipynb
│   ├── test_images_30fps.ipynb
│   ├── test_profiles_1fps.ipynb
│   └── test_profiles_30fps.ipynb
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

## Prerequisites

- **Python 3.8+**
- **pip** package manager
- **Virtual Environment** (recommended)

## Setup and Installation

1. **Clone the repository** (if applicable) or navigate to the project root.

2. **Download Dataset**:
   The dataset is hosted externally due to its size (28GB).
   - **Link**: [Download Dataset](https://cloud.cps.unileoben.ac.at/index.php/s/QA5zC5AAcb6gKH8)
   - **Action**: Extract the contents into the project root.
   - **Structure**: Ensure the folder is named exactly `NNATT dataset`.

3. **Environment Setup**:
   It is highly recommended to use a virtual environment to manage dependencies.

   ```bash
   # Create a virtual environment
   python -m venv venv

   # Activate the environment
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

4. **Install Dependencies**:
   Install the required Python packages from the requirements file.
   ```bash
   pip install -r requirements.txt
   ```
   > **Note for GPU Users**: If you are using NVIDIA GPUs (CUDA) or Apple Silicon (MPS), please ensure you have the appropriate PyTorch version installed for hardware acceleration. Visit [pytorch.org](https://pytorch.org/get-started/locally/) for specific commands.

## How to Run

The project is structured into Jupyter Notebooks located in the `scripts/` directory.

1. **Start Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```
   or open the folder in VS Code and select the appropriate kernel.

2. **Train Models**:
   Navigate to the `scripts/` folder and run the desired notebook matching your data type and frame rate:
   - `images 1fps.ipynb`: Train image classifiers (ResNet, MobileViT, RF) on 1fps data.
   - `images 30fps.ipynb`: Train image classifiers on 30fps data.
   - `profiles 1fps.ipynb`: Train profile classifiers (1D CNN, MLP, RF) on 1fps CSV data.
   - `profiles 30fps.ipynb`: Train profile classifiers on 30fps CSV data.

   Run all cells in the notebook. Key outputs will be displayed inline.

3. **Inference**:
   Use the notebooks in the `test/` directory to load saved models and perform inference on test sets.

## Model Architectures

This project explores a variety of state-of-the-art and classical machine learning models tailored to different data modalities:

### Image Classifiers
- **MobileViT-XS**: A light-weight vision transformer designed for mobile devices. It combines the strengths of CNNs and ViTs to achieve high accuracy with low latency.
- **ResNet50**: A deep residual network that is a standard benchmark in computer vision tasks.
- **Random Forest**: Used as a baseline, operating on flattened image pixel vectors.

### Profile Classifiers (Time-Series/Spectral Data)
- **1D-CNN**: A one-dimensional Convolutional Neural Network optimized for extracting features from sequential profile data.
- **Multilayer Perceptron (MLP)**: A fully connected dense neural network.
- **Random Forest**: An ensemble method that proved highly effective for the structured nature of the profile data.

## Results

After execution, the notebooks generate the following artifacts:

- **Saved Models**: Stored in the `models/` directory.
  - Image models: `model_resnet50_*.pth`, `model_mobilevit_*.pth`, `model_rf_*.joblib`
  - Profile models: `model_cnn_*.pth`, `model_mlp_*.pth`, `model_rf_*.joblib`

- **Visualizations**: Stored in the `results/` directory.
  - **Confusion Matrices**: Heatmaps showing classification performance per class (e.g., `confusion_matrix_images_30fps.png`).
  - **Accuracy Comparisons**: Bar charts comparing different model architectures (e.g., `accuracy_comparison_profiles_1fps.png`).

- **Console Output**:
  - Detailed classification reports (Precision, Recall, F1-Score).
  - Accuracy metrics for each model.
  - Identification of the best performing model.

### Performance Summary

| Data Type | FPS | Model | Accuracy | Training Time | Memory Usage |
|-----------|-----|-------|----------|---------------|--------------|
| Image | 30 | **MobileViT-XS** | **100%** | Fast | Low |
| Image | 30 | ResNet50 | 100% | Moderate | High |
| Image | 30 | Random Forest | 100% | Moderate | High |
| Image | 1 | **MobileViT-XS** | **100%** | Fast | Low |
| Image | 1 | ResNet50 | 100% | Moderate | High |
| Image | 1 | Random Forest | 100% | Moderate | High |
| Profile | 30 | **Random Forest** | **~96.3%** | Instant | Very Low |
| Profile | 30 | **MLP** | **~95.7%** | Very Fast | Very Low |
| Profile | 30 | 1D-CNN | ~85.4% | Very Fast | Very Low |
| Profile | 1 | **Random Forest** | **~99.9%** | Instant | Very Low |
| Profile | 1 | **MLP** | **~99.7%** | Very Fast | Very Low |
| Profile | 1 | 1D-CNN | ~99.5% | Very Fast | Very Low |

### Key Takeaways

- **Images 30fps (Winner: MobileViT-XS)**:
  All architectures achieved **100% accuracy**, demonstrating that the 30fps image dataset contains very distinct features for each class. **MobileViT** takes the top spot because it balances perfect accuracy with mobile-friendly efficiency, unlike ResNet50 (heavy) or Random Forest (high RAM usage for flattening).

- **Images 1fps (Winner: MobileViT-XS)**:
  Similarly, all models scored **100% accuracy** on the 1fps dataset. Visual distinctiveness remains high even at lower frame rates. **MobileViT** remains the preferred choice for its modern architecture, offering spatial invariance that simpler methods lack, while staying lightweight.

- **Profiles 30fps (Winner: Random Forest)**:
  On the 30fps vector profile data, **Random Forest** leads with **~96.3% accuracy**, followed by **MLP (~95.7%)**. The **1D-CNN** trails at **~85.4%**.
  - **Random Forest**: Achieves the highest accuracy with instant training.
  - The drop in CNN accuracy suggests spatial features are less critical or harder to learn at this frame rate compared to the statistical features leveraged by Random Forest.
  
- **Profiles 1fps (Winners: Random Forest & MLP)**:
  Reducing the frame rate to 1fps significantly improved results across all models. **Random Forest (~99.9%)** and **MLP (~99.7%)** are virtually perfect, with the **1D-CNN (~99.5%)** also showing immense improvement. This suggests that the 1fps profile data likely contains cleaner or more distinct signal patterns that all models can exploit effectively.
