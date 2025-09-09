# **AI-Powered-Pneumonia-Detection-in-Low-Resource-Settings**
**Aim:** Develop a **computationally efficient** and **interpretable** deep learning model for pneumonia detection in chest X-rays, optimised for deployment in **low-resource clinical settings**.

## **Abstract**
Pneumonia remains the leading infectious cause of mortality among children under five, claiming over 700,000 lives annually. This research addresses the critical gap between advanced deep learning pneumonia detection capabilities and practical deployment limitations in low-resource clinical environments.

<img width="610" height="607" alt="Pnumonia Unicef" src="https://github.com/user-attachments/assets/04a140e4-e328-46ac-acf9-3ce8a43b5bc0" />

**Image source:** *UNICEF, "A child dies of Pneumonia every 43 seconds globally," Instagram, Nov. 17, 2024. [Online]. Available: https://www.instagram.com/p/DCd4KmqI3bg/ [Accessed September 9, 2025].*

--------
----------

#### Research Design and Overview
<img width="3461" height="3154" alt="image" src="https://github.com/user-attachments/assets/23b1784b-4eba-4123-912a-d6e4f6855f05" />

*Figure1: Methodology Flowchart*


### **Key Achievements:**
- 97.12% accuracy on paediatric data (Xception)
- 86.10% accuracy on adult data (EfficientNetB4)
- 91% model size reduction through quantisation
- Sub-second inference on edge devices (Raspberry Pi)
- Real-time explainability via optimised Grad-CAM

------

### Datasets
##### NIH ChestX-ray14 Dataset

- Size: 112,120 chest X-ray images from 30,805 patients
- Demographics: Adult population (mixed ages)
- Class Distribution: Severe imbalance (60,412 Normal vs 1,431 Pneumonia)
- Challenge: 42:1 class ratio requiring careful balancing strategies

<img width="2321" height="1126" alt="image" src="https://github.com/user-attachments/assets/b4cecb60-ec1d-4c87-87cc-7ce3d410d4aa" />

*Figure 2: Distribution of 14 Labels in NIH ChestX-ray14 Dataset - Bar chart with annotated counts and pie chart with percentage annotations showing all 14 labels*

<img width="1180" height="590" alt="image" src="https://github.com/user-attachments/assets/b3420741-2f38-45de-a2b1-36ad713cc057" />

*Figure 3: Class Distribution for Pneumonia Detection Task - Bar chart and pie chart highlighting the imbalance (60,412 Normal vs. 1,431 Pneumonia)*

##### Mendeley Paediatric Dataset (Kermany et al.)
- Size: 5,856 chest X-ray images
- Demographics: Children aged 1-5 years
- Class Distribution: 4,672 Pneumonia vs 1,184 Normal cases
- Advantage: Higher pneumonia prevalence (79.8%) for robust training
  
<img width="1589" height="617" alt="image" src="https://github.com/user-attachments/assets/e019fae1-abe7-4079-8115-a8a09523ed74" />

*Figure 4: Distribution of Labeled Chest X-Ray Images (Mendeley Data) - Bar chart with counts and pie chart with percentages for Normal and Pneumonia classes*

-----

## Model Performance Results

### Mendeley Dataset (Paediatric) - Top Performers

| Model | Accuracy (%) | Sensitivity (%) | Specificity (%) | AUROC | Size (MB) |
|-------|--------------|-----------------|-----------------|--------|-----------|
| Xception | 96.96 | 96.92 | 97.01 | 0.9933 | 82.08 |
| Nuclear Ensemble | 96.79 | 96.67 | 97.01 | 0.9915 | 300.74 |
| EfficientNetB4 | 95.19 | 97.95 | 90.60 | 0.9887 | 208.32 |
| SE-MobileNetV2 | 94.07 | 95.90 | 91.03 | 0.9851 | 15.27 |

### NIH Dataset (Adult) - Top Performers

| Model | Accuracy (%) | Sensitivity (%) | Specificity (%) | AUROC | Size (MB) |
|-------|--------------|-----------------|-----------------|--------|-----------|
| EfficientNetB4 | 86.10 | 78.91 | 88.99 | 0.8986 | 208.32 |
| Nuclear Ensemble | 78.70 | 38.67 | 94.81 | 0.8528 | 300.74 |
| Xception | 79.54 | 42.27 | 92.53 | 0.8436 | 82.06 |
| VGG16 | 75.56 | 60.74 | 81.53 | 0.7949 | 75.74 |

*CheXNet Baseline: 0.7679 AUROC (EfficientNetB4 achieved 17% improvement)*

<img width="1180" height="790" alt="AUCs Across dataset and models" src="https://github.com/user-attachments/assets/ef986bea-affe-40b1-81c5-ce5068de6898" />

*Figure 5: Comparative AUROC for All Models Across Datasets with CheXNet AUROC as dashed benchmark*

--------

## Novel Architecture Contributions
**1. SE-MobileNetV2**
- Enhanced MobileNetV2 with Squeeze-and-Excitation blocks
- **Parameters: 2.79M** | **Performance:** 94.07% accuracy (Mendeley)
- Adaptive SE integration maintaining computational efficiency
<img width="7109" height="1167" alt="SE-MobileNetV2 Arch" src="https://github.com/user-attachments/assets/aa22af0f-4dc2-4aed-a8cf-a63d029f981e" />

*Figure 6: SE-MobileNetV2 Architecture*

**2. SE-EfficientNetB3**
- Improved EfficientNetB3 with advanced SE mechanisms
- **Parameters: 11.48M** | **Performance:** 92.47% accuracy (Mendeley)
- Dual batch normalisation with progressive dropout
<img width="7109" height="1167" alt="SE-EfficientNetB3 Arch" src="https://github.com/user-attachments/assets/d4253910-464f-4edd-9cdb-72dd22ff3144" />

*Figure 7: SE-EfficientNetB3 Architecture*

**3. PneumoLiteCovNet-25**
- Custom lightweight architecture with multi-scale feature extraction
- **Parameters: 2.49M** | **Performance:** 82.85% accuracy (Mendeley)
- Parallel convolution branches (1×1, 3×3, 5×5 features)
<img width="8970" height="2458" alt="PneumoLitecovNet-25 Arch" src="https://github.com/user-attachments/assets/d863c7ac-3282-44c1-b998-b4c3241e2f1e" />

*Figure 8: PneumoLiteCovNet-25 Architecture*

**4. Nuclear Ensemble**
- Meta-learning ensemble of 5 architectures with weighted voting
- **Components:** ResNet50, DenseNet121, InceptionV3, Xception, MobileNetV2
- **Performance:** 96.79% accuracy with uncertainty quantification
<img width="15950" height="4761" alt="Nuclear Ensemble Model Arch" src="https://github.com/user-attachments/assets/e7703f62-7bf1-4ca9-985d-1c7f771f0373" />

*Figure 9: Nuclear Ensemble Architecture*

-----------

## Quantisation Results

Post-training quantisation achieved significant model compression whilst preserving clinical performance:

| Model | Original Size | Quantised Size | Size Reduction | Accuracy Change |
|-------|---------------|----------------|----------------|-----------------|
| Xception | 82.08 MB | 21.61 MB | 74% | 96.96% → 97.12% |
| Nuclear Ensemble | 300.74 MB | 77.17 MB | 74% | 96.79% → 95.99% |
| EfficientNetB4 | 208.32 MB | 18.59 MB | 91% | 86.10% → 80.94% |
<img width="1430" height="1181" alt="Quantisation Impact Analysis" src="https://github.com/user-attachments/assets/54a4c2e6-3770-4ee4-8961-3ee67fe495c8" />

*Figure 10: Quantisation Impact Analysis - Comparing original versus quantised model performance across key metrics*

-----

## Edge Deployment Performance

Inference performance on Raspberry Pi 4B simulation:

| Model | Dataset | Size (MB) | Inference Time (ms) |
|-------|---------|-----------|---------------------|
| Xception (Quantised) | Mendeley | 21.61 | 550.76 ± 119.67 |
| EfficientNetB4 (Quantised) | NIH | 18.59 | 232.69 ± 93.50 |
| Nuclear Ensemble (Quantised) | Both | 77.17 | 1646.77 ± 409.74 |
<img width="1003" height="904" alt="image" src="https://github.com/user-attachments/assets/40a86644-39b1-48a3-9bd0-bed48738edd1" />

*Figure 11: Edge deployment workflow*

----

## Clinical Explainability

Gradient-weighted Class Activation Mapping (Grad-CAM) provides visual explanations for clinical decision support:

- **Real-time generation**: 5-15× computational speedup achieved
- **Clinical integration**: Attention maps highlight pneumonia regions  
- **Physician trust**: Visual validation of model predictions
<img width="1790" height="506" alt="optimised gradcam Xception Mendeley" src="https://github.com/user-attachments/assets/207abfab-3929-4e90-a48d-39d959f13631" />

*Figure 12: Grad-CAM Attention Visualisations - Xception (Mendeley Dataset) - Demonstrating model attention regions for pneumonia detection in paediatric chest X-rays*

<img width="1790" height="506" alt="optimised gradcam efficeintnetb4" src="https://github.com/user-attachments/assets/bf10c51d-8027-409b-9e15-d9e43681a94e" />

*Figure 13: Grad-CAM Attention Visualisations - EfficientNetB4 (NIH Dataset) - Showing interpretability maps for adult chest X-ray analysis*

-------

## Optimal Deployment Recommendations

### For Paediatric Settings
**Recommended Model**: Xception (Quantised)

- **Accuracy**: 97.12%
- **Size**: 21.61 MB
- **Inference**: 550.76 ms
- **Advantage**: Superior sensitivity (97.69%) for childhood screening

### For Adult/General Settings
**Recommended Model**: EfficientNetB4 (Quantised)

- **Accuracy**: 80.94% (exceeds CheXNet by 17%)
- **Size**: 18.59 MB
- **Inference**: 232.69 ms
- **Advantage**: Balanced sensitivity-specificity for general screening
<img width="741" height="590" alt="image" src="https://github.com/user-attachments/assets/266f8caa-0c59-4eac-9e68-dcce63799490" />

*Figure 14: Confusion Matrix Analysis - Xception (Mendeley Dataset)*

<img width="532" height="497" alt="image" src="https://github.com/user-attachments/assets/fb399595-9567-4f4f-ad03-e561ee387f3f" />

*Figure 15: Confusion Matrix Analysis - EfficientNetB4 (NIH Dataset)*


----

## Clinical Impact

This research addresses UN Sustainable Development Goal 3.2.1 by enabling:

- **Healthcare democratisation**: AI deployment in underserved regions
- **Diagnostic equity**: Quality healthcare access regardless of infrastructure
- **Point-of-care diagnosis**: Sub-second pneumonia detection on edge devices
- **Clinical confidence**: Explainable AI supporting physician decision-making

**Target Impact**: Reducing the 700,000+ annual childhood pneumonia deaths through accessible diagnostic technology.

-----

## Key Findings

- **Cross-dataset performance variation**: 15-20% accuracy differences between paediatric and adult populations
- **Quantisation effectiveness**: Architecture-dependent, with Xception showing improvement post-quantisation
- **Edge deployment feasibility**: Sub-second inference achieved on resource-constrained hardware
- **Clinical explainability**: Real-time Grad-CAM generation enables clinical adoption

-------


## Contributors

- **Evans Konadu** - *Primary Researcher* - [GitHub](https://github.com/EvansKonadu)
- **Prof. Dimitrios Makris** - *Supervisor*

----

## Acknowledgements

- Kingston University School of Computing and Mathematics
- NIH Clinical Center for ChestX-ray14 dataset
- Kermany et al. for Mendeley paediatric dataset


***Contact:** For technical support or clinical collaboration opportunities, please open an issue or contact the research team.*

------
------

## Citation

If you use this work, please cite:
```bibtex
@mastersthesis{konadu2025pneumonia,
  title={AI-Powered Pneumonia Detection in Low Resource Settings},
  author={Konadu, Evans},
  school={Kingston University},
  year={2025},
  type={{MSc} Dissertation},
  supervisor={Makris, Dimitrios}
}

