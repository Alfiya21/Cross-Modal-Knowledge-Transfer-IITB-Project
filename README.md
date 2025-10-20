# Cross-Modal-Knowledge-Transfer-IITB-Project
# T1_G21_ByteBuzz

## Project Overview
Track: Educational Data Analysis (EDA)  
Internship:IITB EdTech Internship 2025, with DYPCET  
Group ID:T1_G21  
Group Name: ByteBuzz  
Group Leader:Alfiya Aslam Mulla  
Faculty Mentor:Mrs. Sushama S Takmare  
Department: Data Science  


## Assigned Problem
Problem ID: 15 — Cross-Modal Knowledge Transfer  

# Objective
Use EEG to train a model and test if eye-tracking or GSR-only models can approximate it (domain adaptation or modality dropout).

# Advanced Objective
Implement adversarial domain adaptation or contrastive learning.

## Problem Workflow

### Step 1: Understand and Prepare the Data
#### 1.1 Identify Target and Inputs
- Target:Task accuracy (binary: Correct/Incorrect) or engagement level (from `PSY.csv`)
- Teacher Modality:EEG (`EEG.csv`)
- Student Modalities:Eye-tracking (`EYE.csv`, `IVT.csv`), GSR (`GSR.csv`), and Facial expressions (`TIVA.csv`)

#### 1.2 Data Organization Strategy
- Per-Trial Basis:Synchronize features from all four modalities per trial.  
- Teacher-Student Pairs:Pair teacher (EEG) and student (Eye-tracking/GSR) features from the same trial.



### Step 2: Preprocessing Pipeline
#### 2.1 Feature Extraction
- EEG: Mean and variance of Delta, Theta, Alpha, Beta, Gamma bands  
- Eye-tracking: Average fixation duration, saccade amplitude, mean pupil size  
- GSR: Mean conductance, slopes, recovery rates  
- Facial Expressions: Average AU intensities or emotion probabilities  

#### 2.2 Feature Alignment
- **Normalization:** z-score normalization for consistent scaling  
- **Dimensionality Reduction:** PCA for comparable lower-dimensional representations  

#### 2.3 Label Encoding
- Encode target as:
  - Correct = 1, Incorrect = 0  
  - Engagement: Low = 0, Medium = 1, High = 2  


### Step 3: Modeling Approaches
#### 3.1 Baseline (Single-Modality Models)
- Train teacher model (e.g., XGBoost) on EEG  
- Train student models on Eye, GSR, and Facial data individually  
- Compare metrics (Accuracy, F1-score) for baseline performance  

#### 3.2 Knowledge Transfer (Teacher → Student)
- Use **knowledge distillation**:
  - EEG model acts as teacher  
  - Student model (e.g., Eye-tracking) learns using both hard labels and teacher’s soft predictions  
  - Add distillation loss to student training objective  

#### 3.3 Domain Adaptation Approaches
- Adversarial Domain Adaptation:
  - Feature extractor and domain discriminator setup  
  - Feature extractor learns domain-invariant embeddings  
- Contrastive Learning:
  - Learn shared embeddings: pull together features from same trial, push apart features from different trials  


### Step 4: Evaluation & Interpretation
#### 4.1 Metrics
- Compare student vs. teacher Accuracy, F1-score, ROC-AUC  
- Evaluate domain alignment via KL divergence or cosine similarity

#### 4.2 Interpretability
- Identify most predictive features after knowledge transfer  
- Apply SHAP for feature-level interpretability  



### Step 5: Experiment and Improve
- Test modality combinations (EEG ↔ Eye, EEG ↔ GSR, EEG ↔ Facial)  
- Apply modality dropout during multimodal model training  
- Explore pretraining and fine-tuning strategies  
- Compare classic distillation vs. adversarial and contrastive learning  


## File & Code Organization

