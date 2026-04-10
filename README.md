# Cross-Modal Knowledge Transfer for Emotion & Engagement Recognition
### IIT Bombay Affective Computing Research Lab | Research Internship (Jun–Sep 2025)

> **Intern:** Part of Team ByteBuzz (T1_G21) — IITB EdTech Internship 2025 with DYPCET  
> **Track:** Educational Data Analysis (EDA) | **Problem ID:** 15  
> **Faculty Mentor:** Mrs. Sushama S. Takmare | **Department:** Data Science

---

## Overview

This project investigates whether lightweight physiological modalities (eye-tracking, GSR) can approximate the predictive power of EEG for student emotion and engagement recognition — using **cross-modal knowledge distillation** and **domain adaptation** techniques.

The core research question: *Can a student model trained on cheaper, less intrusive signals (eye-tracking or GSR) learn to replicate the richer representations learned by a teacher model trained on EEG?*

**Result:** Achieved **77.24% classification accuracy** and **0.6133 macro F1 score** on benchmark datasets using a teacher-student knowledge distillation framework.

---

## Key Results

| Model | Modality | Accuracy | Macro F1 |
|-------|----------|----------|----------|
| Teacher (Baseline) | EEG | — | — |
| Student (KD) | Eye-tracking / GSR | **0.7724** | **0.6133** |

- **Feature space:** 54-dimensional statistical feature vectors from multimodal physiological signals
- **Signals used:** EEG, Galvanic Skin Response (GSR), Eye-tracking, Facial Expressions
- **Framework:** Teacher-Student Knowledge Distillation (soft label + hard label training)

---

## Technical Architecture

### Modalities & Data Sources

| Signal | File | Features Extracted |
|--------|------|--------------------|
| EEG (Teacher) | `EEG.csv` | Mean & variance of Delta, Theta, Alpha, Beta, Gamma bands |
| Eye-tracking (Student) | `EYE.csv`, `IVT.csv` | Fixation duration, saccade amplitude, pupil size |
| GSR (Student) | `GSR.csv` | Mean conductance, slopes, recovery rates |
| Facial Expressions (Student) | `TIVA.csv` | AU intensities, emotion probabilities |
| Labels | `PSY.csv` | Task accuracy (binary), engagement level (3-class) |

### Pipeline

```
Raw Multimodal Data
        │
        ▼
Feature Extraction (per trial, per modality)
        │
        ▼
Preprocessing: z-score normalization → PCA (dimensionality reduction)
        │
        ▼
Label Encoding: Correct/Incorrect (binary) | Low/Medium/High engagement
        │
   ┌────┴─────┐
   ▼          ▼
Teacher     Student
(EEG)    (Eye / GSR)
   │          │
   └──► Knowledge Distillation (soft labels + hard labels)
              │
              ▼
        Evaluation: Accuracy, F1, ROC-AUC, KL Divergence
              │
              ▼
        SHAP Interpretability
```

---

## Methodology

### 1. Baseline — Single-Modality Models
- Trained an **XGBoost teacher** on EEG features
- Trained standalone student models on Eye-tracking, GSR, and Facial data
- Established benchmark accuracy and F1 scores across modalities

### 2. Knowledge Distillation (Teacher → Student)
- EEG model generates **soft probability distributions** (soft labels) over classes
- Student model trained on a combined loss:
  - **Hard label loss** (ground truth)
  - **Distillation loss** (KL divergence from teacher's soft outputs)
- Enables the student model to learn richer decision boundaries from the teacher

### 3. Domain Adaptation (Advanced Objectives)
- **Adversarial Domain Adaptation:** Feature extractor + domain discriminator to learn modality-invariant embeddings
- **Contrastive Learning:** Pull together embeddings from the same trial across modalities; push apart embeddings from different trials

### 4. Evaluation & Interpretability
- Metrics: Accuracy, Macro F1, ROC-AUC
- Alignment: KL divergence, cosine similarity between teacher & student feature spaces
- Interpretability: **SHAP values** to identify most predictive features post-distillation

---

## Experiments

| Experiment | Description |
|------------|-------------|
| EEG ↔ Eye-tracking | Distillation from EEG teacher to eye-tracking student |
| EEG ↔ GSR | Distillation from EEG teacher to GSR student |
| EEG ↔ Facial | Distillation from EEG teacher to facial expression student |
| Modality Dropout | Randomly drop modalities during multimodal training for robustness |
| Classic KD vs. Adversarial | Compare standard distillation against adversarial domain adaptation |
| Pretraining + Fine-tuning | Explore transfer learning strategies across modalities |

---

## Tech Stack

- **Languages:** Python
- **ML Libraries:** XGBoost, scikit-learn, PyTorch / TensorFlow
- **Signal Processing:** NumPy, SciPy (band-power extraction, filtering)
- **Interpretability:** SHAP
- **Visualization:** Matplotlib, Seaborn
- **Dimensionality Reduction:** PCA (scikit-learn)

---

## Skills Demonstrated

- Multimodal physiological signal processing (EEG, GSR, eye-tracking)
- Knowledge distillation and model compression
- Domain adaptation (adversarial training, contrastive learning)
- Feature engineering (54-dimensional statistical vectors)
- Deep learning model optimization
- Statistical analysis and experimental reporting
- Affective computing and educational data analysis

---

## Project Info

| Field | Details |
|-------|---------|
| Internship | IITB EdTech Internship 2025 |
| Host Institution | Indian Institute of Technology (IIT) Bombay — Affective Computing Research Lab |
| Collaborating Institute | DYPCET |
| Group | T1_G21 — ByteBuzz |
| Group Leader | Alfiya Aslam Mulla |
| Mentor | Mrs. Sushama S. Takmare |
| Track | Educational Data Analysis (EDA) |
| Duration | June 2025 – September 2025 |
