# Fatigue Detection using Physiological Signals and Deep Learning

## Overview

This project implements a fatigue detection system using physiological
signal analysis and machine learning. The pipeline processes signals,
extracts statistical and spectral features, trains classification
models, and explains predictions using explainable AI techniques.

The system includes: 
- Signal preprocessing 
- Feature extraction
- Deep learning classification 
- Model explainability using SHAP and LIME

Applications include industrial worker monitoring, wearable sensing
platforms, and human performance analysis.

------------------------------------------------------------------------

## Project Structure

fatigue-detection-ml/
\
├── data/                
├── notebooks/           
│   └── DAN_notebook.ipynb\
├── src/                 
│   ├── preprocessing.py\
│   ├── feature_extraction.py\
│   ├── train_model.py\
│   └── explainability.py\
├── results             
├── requirements.txt\
├── run_pipeline.py\
└── README.md

------------------------------------------------------------------------

## Installation

Clone the repository:

git clone https://github.com/R1SHABHRAJ/fatigue-detection-dan.git 
cd fatigue-detection-ml

Create a virtual environment:

python -m venv venv

Activate the environment:

*  Linux / Mac `source venv/bin/activate`

* Windows `venv\Scripts\activate`

Install dependencies: `pip install -r requirements.txt`

------------------------------------------------------------------------

## Dataset Description

This dataset contains multiparameter recordings collected from industrial workers, including both white‑collar and blue‑collar participants, using a custom‑built device designed specifically for this study.  

The primary objective of data collection was to investigate the feasibility of automated fatigue recognition based on physiological signals. The recorded modalities include:

- **ECG** (Electrocardiogram, measured on the chest lead)  
- **EEG** (Electroencephalogram, measured using frontal electrodes)  
- **GSR** (Galvanic Skin Response, measured on the finger)  

In addition to physiological signals, subjective fatigue assessment was performed using the **Multidimensional Fatigue Inventory (MFI‑20) questionnaire**, providing validated self‑report measures across multiple fatigue dimensions.  



------------------------------------------------------------------------

## Signal Processing

The preprocessing pipeline includes: 
- Bandpass filtering 
- Notch filtering 
- Per subject outlier capping
- Noise removal 
- Signal normalization


------------------------------------------------------------------------

## Feature Extraction

# Feature Extraction

To effectively represent the physiological signals, a set of **23 handcrafted features** were extracted from the processed signals. These features capture both statistical characteristics and signal variability, which are important indicators for fatigue detection.

The extracted features are divided into **time-domain statistical features**.

## Extracted Features

The following 23 features were computed for each signal segment:

1. Mean  
2. Standard Deviation  
3. Variance  
4. Median  
5. Minimum Value  
6. Maximum Value  
7. Range  
8. Interquartile Range (IQR)  
9. Root Mean Square (RMS)  
10. Skewness  
11. Kurtosis  
12. Energy  
13. Entropy  
14. Signal Magnitude Area (SMA)  
15. Mean Absolute Value (MAV)  
16. Zero Crossing Rate  
17. Peak-to-Peak Value  
18. Crest Factor  
19. Shape Factor  
20. Impulse Factor  
21. Margin Factor  
22. Signal Power  
23. Standard Error of the Mean

These features provide a comprehensive representation of the physiological signals by capturing:

- Central tendency of the signal
- Signal variability and dispersion
- Distribution characteristics
- Signal energy and amplitude variations

The resulting **23-dimensional feature vector** is then used as input to the machine learning models, including the **Deep Adaptation Network (DAN)** and the baseline classifiers.

---
# Dimensionality Reduction using PCA

After extracting the **23 handcrafted features**, Principal Component Analysis (PCA) was applied to reduce feature dimensionality and remove redundancy.

Physiological signal features often contain **correlated information**, which can increase model complexity and reduce generalization. PCA transforms the original features into a new set of **uncorrelated variables called principal components**.

---

## Principle of PCA

PCA projects the original feature space into a new coordinate system such that:

- The **first principal component** captures the maximum variance in the data
- Each subsequent component captures the next highest variance while remaining orthogonal to the previous components

Mathematically, PCA transforms the feature matrix **X** as:

X_pca = XW

Where:

- **X** = original feature matrix (23 features)
- **W** = eigenvector matrix
- **X_pca** = transformed feature representation

The eigenvectors are obtained from the **covariance matrix of the standardized features**.

---

## PCA Implementation Steps

The PCA process in this project follows these steps:

1. Standardize the extracted 23 features
2. Compute the covariance matrix
3. Calculate eigenvalues and eigenvectors
4. Sort components by descending eigenvalues
5. Select top principal components
6. Transform the dataset into the reduced feature space

---

## Advantages of Using PCA

Applying PCA provides several benefits:

- Reduces feature dimensionality
- Removes redundant and correlated features
- Improves model training efficiency
- Reduces risk of overfitting
- Enhances generalization performance

---

## Integration with DAN Model

The PCA-transformed feature vectors are used as input to the machine learning models including:

- Deep Adaptation Network (DAN)
- BiLSTM
- BiRNN
- Support Vector Machine (SVM)
- Decision Tree

By reducing feature redundancy, PCA helps the DAN model focus on **the most informative components for fatigue classification**.

------------------------------------------------------------------------
# Deep Adaptation Network (DAN) for Fatigue Detection

## Overview

This project implements a **Deep Adaptation Network (DAN)** for fatigue detection using physiological signal features. The model aims to reduce the distribution mismatch between datasets and improve classification performance.

The proposed system includes:

- Feature extraction from physiological signals
- Domain adaptation using Deep Adaptation Network (DAN)
- Classification of fatigue levels
- Comparison with baseline machine learning models

---

# Deep Adaptation Network (DAN)

## Motivation

Physiological signals collected from different subjects or environments often have **distribution differences**, commonly referred to as **domain shift**.

Traditional machine learning models assume that:

P(source) = P(target)

However, in real-world datasets:

P(source) ≠ P(target)

The **Deep Adaptation Network (DAN)** addresses this problem by learning **domain-invariant features** that align the source and target distributions.

---

# DAN Architecture

The DAN model consists of three main components:

1. Feature extraction layers  
2. Domain adaptation layer  
3. Classification layer  

---

# Feature Extraction

The feature extractor learns high-level representations of the input data.

In our implementation, the network consists of fully connected layers with ReLU activation.

Example architecture:


These layers transform the raw features into a **latent feature space** used for classification.

---

# Domain Adaptation using MMD

To reduce the distribution gap between domains, DAN introduces **Maximum Mean Discrepancy (MMD)** loss.

The MMD measures the difference between two distributions in feature space.

MMD formulation:

MMD²(P,Q) = || (1/n) Σ φ(x_source) − (1/m) Σ φ(x_target) ||²

Where:

- x_source = source domain samples  
- x_target = target domain samples  
- φ(x) = feature mapping function  

The overall loss function is:

L = L_classification + λ × L_MMD

Where:

- L_classification = cross entropy loss  
- L_MMD = domain adaptation loss  
- λ = balancing parameter

This encourages the model to learn **features that are both discriminative and domain invariant**.

---

# Training Procedure

The DAN model is trained using the following steps:

1. Extract features from the input dataset
2. Pass features through the neural network
3. Compute classification loss on labeled data
4. Compute MMD loss between source and target features
5. Update model parameters using backpropagation

The network is trained using the **Adam optimizer**.

Typical parameters:

- Learning Rate: 0.001  
- Batch Size: 32  
- Epochs: 100–200  

---

# Advantages of DAN

The DAN model provides several advantages:

- Handles domain shift effectively  
- Learns transferable representations  
- Improves generalization across datasets  
- Suitable for cross-subject physiological data

---

# Baseline Models

To evaluate the effectiveness of the DAN model, several traditional machine learning models were used as baselines.

## Support Vector Machine (SVM)

Support Vector Machine is a supervised classification algorithm that finds the optimal hyperplane separating different classes. In this project, an **RBF kernel SVM** was used.

## Decision Tree

Decision Tree is a simple classification model that splits the data using feature thresholds. It is easy to interpret but may suffer from overfitting.

## BiRNN (Bidirectional Recurrent Neural Network)

Bidirectional Recurrent Neural Networks process sequential data in both forward and backward directions, allowing the model to capture contextual information from past and future time steps. This improves the representation of temporal dependencies in sequential physiological signals.

## BiLSTM (Bidirectional Long Short-Term Memory)

Bidirectional LSTM extends traditional LSTM by processing sequences in both directions, enabling the network to learn long-term dependencies while utilizing information from both past and future states. This architecture is particularly effective for modeling complex temporal patterns in physiological signal data.

---

# Model Evaluation

The models were evaluated using the following metrics:

- Accuracy  
- Precision  
- Recall  
- F1 Score  
- Confusion Matrix

These metrics were used to compare the performance of **DAN with baseline models**.

---

# Expected Outcome

The Deep Adaptation Network is expected to outperform traditional machine learning models because it reduces domain discrepancy and learns more generalized features.

------------------------------------------------------------------------

# Explainable AI

### SHAP (SHapley Additive Explanations)

Provides global and local feature importance for model predictions.

Outputs include: 
- SHAP summary plots 
- Feature contribution visualization

### LIME (Local Interpretable Model-Agnostic Explanations)

Provides local explanations for individual predictions.

---
# Statistical Significance Testing

To ensure that the improvements achieved by the proposed **Deep Adaptation Network (DAN)** model are not due to random variation, statistical significance testing was performed on the model results.

## Purpose

Machine learning models can sometimes show performance improvements due to randomness in:

- Data splitting
- Model initialization
- Training stochasticity

Therefore, statistical testing is used to determine whether the performance differences between models are **statistically significant**.

## Test Used

The notebook performs a **Wilcoxon Signed-Rank Tes** to compare the performance of the proposed model with baseline models.

The comparison is conducted between:

- DAN vs SVM
- DAN vs Decision Tree
- DAN vs Standalone BiRNN
- DAN vs Standalone BiLSTM

The statistical test evaluates whether the difference in model predictions is significant.
## Wilcoxon Signed-Rank Test (DAN+PCA vs Baselines)

This test evaluates whether **DAN + PCA performs significantly better than each baseline model**.

**Null Hypothesis (H₀):** There is no performance difference between DAN+PCA and the baseline model.  
**Significance Level:** α = 0.05  
**Decision Rule:** Reject H₀ if **p < 0.05**

| Comparison | DAN+PCA Accuracy | Baseline Accuracy | Difference | W-statistic | p-value | Significant |
|-------------|------------------|-------------------|-----------|-------------|---------|-------------|
| DAN+PCA vs Decision Tree | 0.8584 | 0.6758 | 0.1826 | 1219.0 | 0.0000 | YES |
| DAN+PCA vs SVM | 0.8584 | 0.7671 | 0.0913 | 518.0 | 0.0004 | YES |
| DAN+PCA vs DAN Original | 0.8584 | 0.8128 | 0.0457 | 110.5 | 0.0062 | YES |
| DAN+PCA vs BiLSTM | 0.8584 | 0.7945 | 0.0639 | 462.5 | 0.0098 | YES |
| DAN+PCA vs BiRNN | 0.8584 | 0.8082 | 0.0502 | 414.0 | 0.0315 | YES |

**Interpretation:**  
All comparisons show **p-values < 0.05**, indicating that the improvements achieved by **DAN+PCA** over the baseline models are **statistically significant**.

------------------------------------------------------------------------

## Running the Pipeline

Run the main pipeline:

python run_pipeline.py

Or run the notebook:

jupyter notebook notebooks/DAN_notebook.ipynb

------------------------------------------------------------------------

## Results

Outputs include: 
- Classification reports 
- Confusion matrices 
- SHAP plots 
- LIME explanations

These are saved in the results/ directory.

------------------------------------------------------------------------

## Applications

-   Industrial fatigue monitoring
-   Wearable health devices
-   Driver fatigue detection
-   Occupational safety monitoring

------------------------------------------------------------------------

## Future Improvements

Potential extensions: 
- Deep learning-based fatigue detection 
- Real-time fatigue monitoring systems 
- Integration with wearable sensors 
- Multimodal physiological signal fusion

------------------------------------------------------------------------

## License

MIT License

------------------------------------------------------------------------

## Author

Dr. Sanchita Paul\
Dr. Rishabh Raj\
Birla Institute of Technology\
Research Project: Fatigue Detection in Industrial Workers
