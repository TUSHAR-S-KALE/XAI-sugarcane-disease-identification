# XAI-Driven Sugarcane Leaf Disease Detection

An explainable deep learning framework for **accurate and interpretable sugarcane leaf disease classification** using EfficientNet-B0, MobileNetV2, and ResNet50, integrated with Grad-CAM visualizations and real-time deployment interfaces.

---

## Overview

This project presents a robust pipeline for **plant disease detection in precision agriculture**, combining:

- Deep Learning (CNNs)
- Explainable AI (Grad-CAM)
- Data Augmentation for class balancing
- Real-time deployment (Desktop + Mobile UI)

---

## Key Features

- Multi-model comparison (EfficientNet-B0, MobileNetV2, ResNet50)
- Explainability using Grad-CAM
- Balanced dataset using augmentation techniques
- High accuracy (~97%+ across all models)
- Desktop (Tkinter) + Mobile (Kivy) applications
- Client-server architecture for real-time inference

---

## Dataset

- **Total Images:** 6,748
- **Classes:** 12
  - 9 Disease Classes
  - Healthy Leaves
  - Dried Leaves
  - Custom **Insect Damage** class

### Dataset Improvements
- Removed mislabeled Pokkah Boeng samples
- Created new **Insect Damage** class
- Cleaned noisy data

---

## Methodology

```mermaid
flowchart TD

A[Dataset Collection] --> B[Data Cleaning & Refinement]
B --> C[Class Imbalance Handling]
C --> D[Data Augmentation]

D --> E[Dataset Splitting]
E --> F[Train Set]
E --> G[Validation Set]
E --> H[Test Set]

F --> I[Model Training]
I --> J1[EfficientNet-B0]
I --> J2[MobileNetV2]
I --> J3[ResNet50]

J1 --> K[Model Evaluation]
J2 --> K
J3 --> K

K --> L[Performance Metrics]
K --> M[Confusion Matrix]

L --> N[Explainability - Grad-CAM]
M --> N

N --> O[Visualization & Analysis]

O --> P[Deployment]
P --> Q[Desktop UI - Tkinter]
P --> R[Mobile UI - Kivy]
```

---

## Data Preprocessing & Augmentation

To handle class imbalance and improve generalization:

- Random Horizontal Flip
- Brightness Adjustment
- Contrast Variation
- Saturation Changes
- 90° Rotations

### Dataset split
- Train: 70%
- Validation: 20%
- Test: 10%

---

## Dataset Distribution

### Before Balancing
![Before Balancing](Assets/Figures/Dataset_distribution_before_class_balancing.png)

### After Balancing
![After Balancing](Assets/Figures/Balanced_train_valid_test_split.png)

---

## Model Architectures

| Model | Parameters | Key Strength |
|------|----------|-------------|
| EfficientNet-B0 | ~5.3M | Best accuracy & efficiency |
| MobileNetV2 | ~3.4M | Lightweight, mobile-friendly |
| ResNet50 | ~25.6M | Deep feature learning |

---

## Training Configuration

- Pretrained: ImageNet
- Optimizer: AdamW
- LR: 3e-4
- Scheduler: Cosine Annealing Warm Restarts
- Loss: Cross-Entropy (Label smoothing = 0.1)
- Epochs: 50
- Early Stopping: Patience = 8
- Mixed Precision Training (AMP)

---

## Training Performance

### Training & Validation Curves

| Metrics | EfficientNet-B0 | MobileNetV2 | ResNet50 |
|---------|-----------------|-------------|----------|
| Accuracy | ![EfficientNet-B0 Training and Validation Accuracy](Assets/EfficientNetB0/Accuracy.png) | ![MobileNetV2 Training and Validation Accuracy](Assets/MobileNetV2/Accuracy.png) | ![ResNet50 Training and Validation Accuracy](Assets/ResNet50/Accuracy.png) |
| Loss | ![EfficientNet-B0 Training and Validation Accuracy](Assets/EfficientNetB0/Loss.png) | ![MobileNetV2 Training and Validation Accuracy](Assets/MobileNetV2/Loss.png) | ![ResNet50 Training and Validation Accuracy](Assets/ResNet50/Loss.png) |
| Precision | ![EfficientNet-B0 Training and Validation Accuracy](Assets/EfficientNetB0/Precision.png) | ![MobileNetV2 Training and Validation Accuracy](Assets/MobileNetV2/Precision.png) | ![ResNet50 Training and Validation Accuracy](Assets/ResNet50/Precision.png) |
| Recall | ![EfficientNet-B0 Training and Validation Accuracy](Assets/EfficientNetB0/Recall.png) | ![MobileNetV2 Training and Validation Accuracy](Assets/MobileNetV2/Recall.png) | ![ResNet50 Training and Validation Accuracy](Assets/ResNet50/Recall.png) |
| F1-Score | ![EfficientNet-B0 Training and Validation Accuracy](Assets/EfficientNetB0/F1_score.png) | ![MobileNetV2 Training and Validation Accuracy](Assets/MobileNetV2/F1_score.png) | ![ResNet50 Training and Validation Accuracy](Assets/ResNet50/F1_score.png) |

---

## Model Performance Comparison

### Bar Plots (Train / Validation / Test)
| EfficientNet-B0 | MobileNetV2 | ResNet50 |
|-----------------|-------------|----------|
| ![Performance Comparison](Assets/EfficientNetB0/train_valid_test_metrics.png) | ![Performance Comparison](Assets/MobileNetV2/train_valid_test_metrics.png) | ![Performance Comparison](Assets/ResNet50/train_valid_test_metrics.png) |

---

## Confusion Matrix

![Confusion Matrix](Assets/Figures/Confusion_matrix.png)

---

## Explainability (Grad-CAM)

Model focuses on **disease-relevant regions** instead of background noise.

![Grad-CAM Samples](Assets/Figures/XAI_comparison.png)

---

## Results Summary

| Model | Accuracy | F1 Score |
|------|--------|---------|
| EfficientNet-B0 | **97.85%** | **97.84%** |
| MobileNetV2 | 97.64% | 97.62% |
| ResNet50 | 97.64% | 97.63% |

---

## Client-Server Architecture

```mermaid
flowchart TD

subgraph Client Side
A[User - Mobile/Desktop] --> B[Upload/Capture Image]
B --> C[Select Model]
end

C --> D[Send Request to Server]

subgraph Server Side
D --> E[Remote Server]
E --> F[Load Model]
F --> G[Perform Prediction]
G --> H[Generate Grad-CAM]
end

H --> I[Return Results]

subgraph Output
I --> J[Display Prediction]
I --> K[Display Grad-CAM]
J --> L[User Interface]
K --> L
end
```

---

## User Interface (Demo/Preview)

### Desktop Application (Tkinter)

**Home Screen**
![Desktop Home](Assets/Figures/Desktop_homescreen.png)

**Result Screen**
![Desktop Result](Assets/Figures/Desktop_result_screen.png)

---

### Mobile Application (Kivy)

**Home Screen**
![Mobile Home](Assets/Figures/Mobile_homescreen.png)

**Result Screen**
![Mobile Result](Assets/Figures/Mobile_result_screen.png)

---

## System Architecture

- Client captures/uploads image
- Sends request to remote server
- Server:
  - Performs prediction
  - Generates Grad-CAM
- Returns result to UI

---

## How to Run

### 1. Start Server
```bash
python server.py
```

### 2. Run Desktop App
```
python desktop_app.py
```

### 3. Run Mobile App
```
python main.py
```

---

## Important Notes

- Server must be running before using applications
- Models are loaded from the backend server
- Ensure correct environment setup

---

## Applications
- Precision Agriculture
- Smart Farming Systems
- Disease Monitoring Tools
- Mobile-based Crop Diagnosis

---

## Conclusion

This project demonstrates that lightweight deep learning models combined with XAI can deliver:

- High accuracy
- Real-time performance
- Interpretability

making it suitable for real-world agricultural deployment.

---

## Tech Stack

- **Languages:** Python  
- **Deep Learning:** PyTorch, Torchvision  
- **Models:** EfficientNet-B0, MobileNetV2, ResNet50  
- **XAI:** Grad-CAM  
- **Data Processing:** NumPy, Pandas, OpenCV  
- **Visualization:** Matplotlib, Seaborn  
- **Deployment:** Tkinter (Desktop), Kivy (Mobile)  
- **Backend:** Client-Server Architecture  
- **Hardware:** NVIDIA GPU (CUDA), Mixed Precision Training

---

## License

This project is licensed under the MIT License.

---