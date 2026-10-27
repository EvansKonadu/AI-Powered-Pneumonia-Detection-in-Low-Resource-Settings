# **AI-Powered-Pneumonia-Detection-in-Low-Resource-Settings**
**Aim:** Develop a **computationally efficient** and **interpretable** deep learning model for pneumonia detection in chest X-rays, optimised for deployment in **low-resource clinical settings**.

> **Note (updated October 2026).** This README describes the MSc dissertation. Its values have been updated to match the journal manuscript *Computationally Efficient AI Models for Pneumonia Detection in Low-Resource Settings*, prepared for submission to *Discover Artificial Intelligence*. The notebooks, the additional experiments and the verification code for that manuscript are in the folder [`discover_ai_paper`](discover_ai_paper/).

## **Abstract**
Pneumonia is the single largest infectious cause of death in children under five years old, and the World Health Organization reports 740,180 deaths in this age group in 2019. This research addresses the gap between advanced deep learning pneumonia detection and the practical limits of low-resource clinical environments, where hardware is modest and an internet connection cannot be assumed.

<img width="610" height="607" alt="Pnumonia Unicef" src="https://github.com/user-attachments/assets/04a140e4-e328-46ac-acf9-3ce8a43b5bc0" />

**Image source:** *UNICEF, "A child dies of Pneumonia every 43 seconds globally," Instagram, Nov. 17, 2024. [Online]. Available: https://www.instagram.com/p/DCd4KmqI3bg/ [Accessed September 9, 2025].*

--------
----------

#### Research Design and Overview
<img width="3461" height="3154" alt="image" src="https://github.com/user-attachments/assets/23b1784b-4eba-4123-912a-d6e4f6855f05" />

*Figure 1: Methodology Flowchart*


### **Key Results:**
- 97.12% accuracy on paediatric data (quantised Xception, 96.96% before quantisation)
- 86.10% accuracy on adult data (EfficientNetB4; 80.94% after dynamic-range quantisation)
- Quantised models were 73 to 74% smaller than their FP32 equivalents (91% smaller than the saved Keras file for EfficientNetB4, which includes optimiser state)
- Inference times of 233 ms (EfficientNetB4) and 551 ms (Xception) under a CPU configuration scaled to a Raspberry Pi 4B. These are clock-scaled estimates and not measurements on the board
- A compiled Grad-CAM implementation that gave identical maps 12 to 14 times faster for EfficientNetB4 and about 5 times faster for Xception

------

### Datasets
##### NIH ChestX-ray14 Dataset

- Size: 112,120 chest X-ray images from 30,805 patients
- Demographics: Adult population (mixed ages)
- Class distribution for the binary task: 60,357 normal (`No Finding`) against 1,431 pneumonia images
- Challenge: a class ratio of about 42 to 1, which required a balanced selection of normal images (5,341 chosen by k-means clustering)

> Figures 2 and 3 were produced for the dissertation and show 60,412 normal images. The journal manuscript uses the counts taken from the released metadata file (60,357).

<img width="2321" height="1126" alt="image" src="https://github.com/user-attachments/assets/b4cecb60-ec1d-4c87-87cc-7ce3d410d4aa" />

*Figure 2: Distribution of 14 Labels in NIH ChestX-ray14 Dataset - Bar chart with annotated counts and pie chart with percentage annotations showing all 14 labels*

<img width="1180" height="590" alt="image" src="https://github.com/user-attachments/assets/b3420741-2f38-45de-a2b1-36ad713cc057" />

*Figure 3: Class Distribution for Pneumonia Detection Task - Bar chart and pie chart highlighting the imbalance (60,412 normal against 1,431 pneumonia, as counted for the dissertation)*

##### Mendeley Paediatric Dataset (Kermany et al.)
- Size: 5,856 chest X-ray images
- Demographics: Children aged 1-5 years
- Class distribution: 4,273 pneumonia and 1,583 normal images (73.0% pneumonia)
- Partition: the distributed partition was used (training folder of 5,232 images and test folder of 624 images)

> Figure 4 was produced for the dissertation. The class counts above are those of the released dataset and are used in the journal manuscript.

<img width="1589" height="617" alt="image" src="https://github.com/user-attachments/assets/e019fae1-abe7-4079-8115-a8a09523ed74" />

*Figure 4: Distribution of Labeled Chest X-Ray Images (Mendeley Data) - Bar chart with counts and percentages for Normal and Pneumonia classes*

-----

## Model Performance Results

Test sets, threshold 0.5. Sizes are those of the saved model files.

### Mendeley Dataset (Paediatric) - Selected Models

| Model | Accuracy (%) | Sensitivity (%) | Specificity (%) | AUROC | Size (MB) |
|-------|--------------|-----------------|-----------------|--------|-----------|
| Xception | 96.96 | 96.92 | 97.01 | 0.9933 | 82.06 |
| Nuclear Ensemble* | 96.63 | 96.67 | 96.58 | 0.9914 | 300.74 |
| EfficientNetB4 | 95.19 | 97.95 | 90.60 | 0.9890 | 208.44 |
| SE-MobileNetV2 | 94.07 | 95.90 | 91.03 | 0.9851 | 15.27 |

\*Vote weights tuned on the validation images. The weights archived with the original ensemble were tuned on test images and gave 96.79% accuracy and an AUROC of 0.9915.

### NIH Dataset (Adult) - Selected Models

| Model | Accuracy (%) | Sensitivity (%) | Specificity (%) | AUROC | Size (MB) |
|-------|--------------|-----------------|-----------------|--------|-----------|
| EfficientNetB4 | 86.10 | 78.91 | 88.99 | 0.8986 | 208.32 |
| Xception | 79.54 | 47.27 | 92.53 | 0.8436 | 81.58 |
| Nuclear Ensemble | 78.70 | 38.67 | 94.81 | 0.8529 | 505.89 |
| VGG16 | 75.56 | 60.74 | 81.53 | 0.7947 | 75.74 |

*CheXNet reported an AUROC of 0.768 for pneumonia on its own ChestX-ray14 test set. The partitions differ, so the values are not directly comparable. The NIH binary partition used here is at image level.*

<img width="1180" height="790" alt="AUCs Across dataset and models" src="https://github.com/user-attachments/assets/ef986bea-affe-40b1-81c5-ce5068de6898" />

*Figure 5: Comparative AUROC for All Models Across Datasets with CheXNet AUROC as dashed benchmark (different test partitions)*

--------

## Architectural Designs
**1. SE-MobileNetV2**
- MobileNetV2 with Squeeze-and-Excitation blocks
- **Parameters: 2.79M** | **Performance:** 94.07% accuracy (Mendeley)
- Squeeze-and-excitation blocks added with a small increase in parameters
<img width="7109" height="1167" alt="SE-MobileNetV2 Arch" src="https://github.com/user-attachments/assets/aa22af0f-4dc2-4aed-a8cf-a63d029f981e" />

*Figure 6: SE-MobileNetV2 Architecture*

**2. SE-EfficientNetB3**
- EfficientNetB3 with Squeeze-and-Excitation blocks, batch normalisation and additional dropout
- **Parameters: 11.48M** | **Performance:** 92.47% accuracy (Mendeley)
<img width="7109" height="1167" alt="SE-EfficientNetB3 Arch" src="https://github.com/user-attachments/assets/d4253910-464f-4edd-9cdb-72dd22ff3144" />

*Figure 7: SE-EfficientNetB3 Architecture*

**3. PneumoLiteCovNet-25**
- Custom lightweight architecture with multi-scale feature extraction
- **Parameters: 2.49M** | **Performance:** 82.69% accuracy (Mendeley)
- Parallel convolution branches (1×1, 3×3, 5×5 features)
<img width="8970" height="2458" alt="PneumoLitecovNet-25 Arch" src="https://github.com/user-attachments/assets/d863c7ac-3282-44c1-b998-b4c3241e2f1e" />

*Figure 8: PneumoLiteCovNet-25 Architecture*

**4. Nuclear Ensemble**
- Ensemble of 5 architectures combined by a neural meta-learner and a weighted vote tuned with Optuna
- **Components:** ResNet50, DenseNet121, InceptionV3, Xception, MobileNetV2
- **Performance:** 96.63% accuracy on the Mendeley test set (validation-tuned weights)
<img width="15950" height="4761" alt="Nuclear Ensemble Model Arch" src="https://github.com/user-attachments/assets/e7703f62-7bf1-4ca9-985d-1c7f771f0373" />

*Figure 9: Nuclear Ensemble Architecture*

-----------

## Quantisation Results

Post-training 8-bit quantisation reduced file sizes substantially. The effect on predictive performance depended on the model:

| Model | Original Size | Quantised Size | Size Reduction | Accuracy Change |
|-------|---------------|----------------|----------------|-----------------|
| Xception (full-integer) | 82.06 MB | 21.61 MB | 74% | 96.96% → 97.12% |
| Nuclear Ensemble (full-integer members) | 300.74 MB | 77.17 MB | 74% | 96.63% → 94.39% |
| EfficientNetB4 (dynamic-range) | 208.32 MB | 18.59 MB | 91%* | 86.10% → 80.94% |

\*Relative to the saved Keras file, which holds 69.17 MB of weights and 137.39 MB of optimiser state. Relative to the FP32 weights the reduction is 73%.

Balanced accuracy was unchanged for Xception (96.97% to 96.84%), fell by 5.5 points for the NIH EfficientNetB4 and by 3.4 points for the ensemble.

<img width="1430" height="1181" alt="Quantisation Impact Analysis" src="https://github.com/user-attachments/assets/54a4c2e6-3770-4ee4-8961-3ee67fe495c8" />

*Figure 10: Quantisation Impact Analysis - Comparing original versus quantised model performance across key metrics*

-----

## Edge Deployment Performance

Inference times were measured on a Google Colab CPU session (2.2 GHz, one physical core) and multiplied by 1.47 to scale the clock to the 1.5 GHz of the Raspberry Pi 4 Model B. This scales the clock only and does not reproduce the board, so the times are estimates of relative cost. Mean ± standard deviation over 50 test images, batch size one.

| Model | Dataset | Size (MB) | Quantised (ms) | FP32 TensorFlow Lite (ms) |
|-------|---------|-----------|----------------|---------------------------|
| Xception | Mendeley | 21.61 | 550.76 ± 119.67 | 378.40 ± 73.40 |
| EfficientNetB4 | NIH | 18.59 | 232.69 ± 93.50 | 100.35 ± 36.34 |
| Nuclear Ensemble (five members) | Mendeley | 77.17 | 1646.77 ± 409.74 | 1591.36 ± 435.70 |

Floating-point TensorFlow Lite files were as fast as or faster than the quantised files on this host. Most of the reduction in time relative to the Keras models therefore came from the TensorFlow Lite runtime and not from quantisation. Quantisation reduced file size, which determines whether a model can be stored on a small device.
<img width="1003" height="904" alt="image" src="https://github.com/user-attachments/assets/40a86644-39b1-48a3-9bd0-bed48738edd1" />

*Figure 11: Edge deployment workflow*

----

## Explainability

Gradient-weighted Class Activation Mapping (Grad-CAM) provides visual explanations of predictions:

- **Computation time**: a compiled TensorFlow implementation gave identical maps 12 to 14 times faster than the standard implementation for EfficientNetB4 and about 5 times faster for Xception (0.3 to 0.7 s per image on the scaled CPU configuration)
- **Input size**: reducing the input to 128 × 128 pixels was faster still but changed the predictions, so it is not a safe optimisation
- **Interpretation**: the maps were not compared with expert annotations. They illustrate the workflow and its cost and are not evidence that the models attend to disease
<img width="1790" height="506" alt="optimised gradcam Xception Mendeley" src="https://github.com/user-attachments/assets/207abfab-3929-4e90-a48d-39d959f13631" />

*Figure 12: Grad-CAM Visualisations - Xception (Mendeley Dataset) - paediatric chest X-rays*

<img width="1790" height="506" alt="optimised gradcam efficeintnetb4" src="https://github.com/user-attachments/assets/bf10c51d-8027-409b-9e15-d9e43681a94e" />

*Figure 13: Grad-CAM Visualisations - EfficientNetB4 (NIH Dataset) - adult chest X-rays*

-------

## Deployment Options

### For Paediatric Settings
**Option**: Xception (full-integer quantised)

- **Accuracy**: 97.12%
- **Size**: 21.61 MB
- **Inference**: 550.76 ms (scaled estimate)
- **Sensitivity**: 97.95% (specificity 95.73%)

### For Adult/General Settings
**Option**: EfficientNetB4 (dynamic-range quantised)

- **Accuracy**: 80.94% (original 86.10%)
- **Size**: 18.59 MB
- **Inference**: 232.69 ms (scaled estimate)
- **Sensitivity and specificity**: 72.66% and 84.28%
<img width="741" height="590" alt="image" src="https://github.com/user-attachments/assets/266f8caa-0c59-4eac-9e68-dcce63799490" />

*Figure 14: Confusion Matrix Analysis - Xception (Mendeley Dataset)*

<img width="532" height="497" alt="image" src="https://github.com/user-attachments/assets/fb399595-9567-4f4f-ad03-e561ee387f3f" />

*Figure 15: Confusion Matrix Analysis - EfficientNetB4 (NIH Dataset)*

These models are research prototypes. No clinical evaluation was made and they are not intended for clinical use.

----

## Context

Pneumonia accounts for 14% of all deaths of children under five years old (WHO, 2019). Where radiology resources are scarce, the use of artificial intelligence for imaging is limited by equipment, staff expertise and infrastructure. This work examines how far compression can reduce the size and computational cost of chest radiograph classifiers, and what it costs in predictive performance.

-----

## Key Findings

- **Cross-dataset performance variation**: every architecture scored higher on the paediatric Mendeley data than on the adult NIH data (accuracy by 9 to 20 percentage points). The datasets differ in age group, hospital, class prevalence and labels, and these results do not separate those factors
- **Quantisation effect depends on the model**: Xception was unchanged within the resolution of its 624-image test set, whereas the NIH EfficientNetB4 lost 5.5 points of balanced accuracy and the ensemble 3.4 points
- **Timing**: the times are clock-scaled estimates. A floating-point TensorFlow Lite baseline in the same runtime should be used when reporting the effect of quantisation, and latency on a physical Raspberry Pi remains to be measured
- **Explainability**: compiling the Grad-CAM computation gave identical maps at a lower cost, whereas lowering the input size changed the predictions

-------


## Contributors

- **Evans Konadu** - *Primary Researcher* - [GitHub](https://github.com/EvansKonadu)
- **Prof. Dimitrios Makris** - *Supervisor*

----

## Acknowledgements

- Kingston University School of Computing and Mathematics
- NIH Clinical Center for ChestX-ray14 dataset
- Kermany et al. for Mendeley paediatric dataset


***Contact:** For technical support or collaboration opportunities, please open an issue or contact the research team.*

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
```
The journal manuscript is described in [`discover_ai_paper`](discover_ai_paper/).
