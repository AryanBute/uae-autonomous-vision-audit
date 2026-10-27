# uae-autonomous-vision-audit
Auditing Vision Transformers (ViT-B/16) for algorithmic bias against Vulnerable Road Users (VRUs) under adverse Middle Eastern weather conditions (Sandstorm/Glare).
# Algorithmic Bias of Vision Transformers on Vulnerable Road Users 🏍️🌪️

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97_Hugging_Face-Models-orange)
![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)

## Abstract
This repository contains the implementation and findings of an AI safety audit investigating algorithmic bias in state-of-the-art autonomous driving vision models. Specifically, it evaluates whether Vision Transformers (ViT-B/16) exhibit disproportionate performance degradation when classifying Vulnerable Road Users (VRUs)—such as motorbike delivery riders—compared to standard commercial vehicles under localized adverse weather conditions (Sandstorm Haze and Gulf Sun Glare).

Trained on the **Emirates Multi-Task (EMT) dataset**, this project proves that while high average accuracy is maintained in clean weather, simulated sandstorms expose a critical, potentially lethal blind spot for VRUs.

## 📊 Key Findings: The Illusion of Average Accuracy
* **Clean Weather Baseline:** 97.23% Validation Accuracy. F1-scores are highly stable across all classes (Cars: 0.94, Motorbikes: 0.92).
* **Gulf Sun Glare:** The self-attention mechanism of the ViT handles brightness shifting effectively, maintaining a 93% overall accuracy.
* **Sandstorm Haze (The Bias):** Under low-contrast blur simulating a UAE sandstorm, algorithmic bias is empirically exposed:
  * Standard Car F1-score drop: **-0.03** (0.94 → 0.91)
  * Motorbike (VRU) F1-score drop: **-0.17** (0.92 → 0.75)
  * *Conclusion:* The model protects massive rigid vehicles but sacrifices unstructured, smaller VRUs.

## 🗄️ Dataset & Preprocessing
This project utilizes the **Emirates Multi-Task (EMT) Dataset** (Madjid et al., 2025), comprising 57 minutes of UAE dashcam footage (Abu Dhabi & Dubai). 

* **Scale:** The raw video extraction pipeline processed ~30 GB of data, expanding to a 110 GB model and environment footprint during training.
* **Temporal Striding:** To prevent data leakage (memorization of static backgrounds across consecutive frames), a **5-frame temporal stride** was applied during extraction.
* **Target Classes:** 
  1. `Standard_Car`
  2. `Large_Commercial` (Trucks/Buses)
  3. `VRU_Motorbike` (Delivery Riders)
  4. `VRU_Pedestrian_Cyclist`

## 🧠 Model Architecture & Training
* **Backbone:** `google/vit-base-patch16-224-in21k`
* **Optimizer:** AdamW (Weight decay effectively prevents overfitting on background noise)
* **Learning Rate:** `3e-5` with a Cosine Annealing schedule and 10% Warmup.
* **Epochs:** 6
* **Hardware:** NVIDIA GPU Accelerated (Google Colab / Local MLOps environment)

## ⚙️ Repository Structure
```text
├── data/                   # Extraction scripts and data loaders (data not included due to size/PII)
├── notebooks/
│   └── MAAI901_ViT_VRU_Bias.ipynb  # Main training, validation, and perturbation loop
├── reports/
│   ├── MAAI901_Methodology.pdf
│   └── MAAI901_Final_Report.pdf
├── requirements.txt        # Python dependencies
└── README.md
