# Computationally Efficient AI Models for Pneumonia Detection in Low-Resource Settings

Code accompanying the manuscript prepared for submission to *Discover Artificial Intelligence* (topical collection "Intelligent Imaging Technologies, Methods and Applications").

**Authors:** Evans Konadu, Negar Abbasi, Siddhesh Thakur, Tariq Rahim, Spyridon Bakas and Dimitrios Makris (corresponding author, Kingston University).

This folder holds the notebooks added for the journal manuscript. The training notebooks for the ten architectures and the Nuclear Ensemble, on both datasets, were written for the MSc dissertation and remain in the repository root under [`notebooks/`](../notebooks/) ([Mendeley](../notebooks/mendeley_dataset/) and [NIH](../notebooks/nih_dataset/)). The root [README](../README.md) describes the dissertation.

## Contents of this folder

| File | Content |
|------|---------|
| `Multi_Label_EfficientNetB4_NIH.ipynb` | 14-label EfficientNetB4 with a multi-scale attention block: patient-level partition, multi-label random over-sampling, training, evaluation and dynamic-range quantisation |
| `Optimisation_NIH_Mendeley.ipynb` | Conversion to TensorFlow Lite (FP32, dynamic-range and full-integer), knowledge distillation into MobileNetV2 students, ensemble evaluation, and inference timing |
| `Verification_and_Sensitivity_Analyses.ipynb` | Checks added during revision: file inspection, FP32 baseline, thread and XNNPACK comparisons, parameters and multiply-accumulate operations, and the Grad-CAM benchmark |

## Where each part of the manuscript comes from

| Manuscript item | Source |
|-----------------|--------|
| Table 4 (nine single networks, both datasets) | Notebooks in [`notebooks/mendeley_dataset`](../notebooks/mendeley_dataset/) and [`notebooks/nih_dataset`](../notebooks/nih_dataset/) |
| Table 5 (Xception, ensemble members, Nuclear Ensemble) | `Nuclear_Ensemble_Classifier_Mendeley.ipynb` and `Nuclear_Ensemble_Classifier_NIH.ipynb` in the same folders |
| Table 6, Fig. 3, Appendix Tables A1 and A4 (quantisation and distillation) | `Optimisation_NIH_Mendeley.ipynb` |
| Table 7 (inference time) | `Optimisation_NIH_Mendeley.ipynb`, with the FP32 baseline in `Verification_and_Sensitivity_Analyses.ipynb` |
| Appendix Tables A5 and A6 (threads, XNNPACK), Table 8 (parameters and MACs) | `Verification_and_Sensitivity_Analyses.ipynb` |
| Table 9 and Fig. 5 (Grad-CAM) | `Verification_and_Sensitivity_Analyses.ipynb` |
| Fig. 4, Appendix Tables A2 and A3 (multi-label) | `Multi_Label_EfficientNetB4_NIH.ipynb` |

## Data

- NIH ChestX-ray14: <https://nihcc.app.box.com/v/ChestXray-NIHCC> (accessed 17 January 2026)
- Mendeley chest X-ray dataset (Kermany et al.): <https://doi.org/10.17632/rscbjbr9sj.3>

The notebooks were run in Google Colab and read the images from Google Drive paths (for example `/content/drive/MyDrive/chest_xray` and `/content/drive/MyDrive/NIH_processed`). Change these paths to the location of the downloaded data before running.

## Dataset partitions

- **Mendeley:** the distributed partition, with 5,232 training and 624 test images. A random 10% of the training folder (523 images) was used for validation for the stand-alone networks and 20% (1,045 images) for the ensemble members and distillation students.
- **NIH binary task:** pneumonia and normal images were split at random at image level (seed 42), giving 1,784 test images (1,272 normal, 512 pneumonia). The training folder holds 5,320 images, including 332 augmented pneumonia images, and 20% of it was used for validation. Absolute figures are specific to this partition and are best read as paired comparisons between an original model and its optimised versions.
- **NIH multi-label task:** a patient-level partition (89,826 training images from 24,644 patients; 22,294 held-out images from 6,161 patients, divided into validation and test halves of 11,147 images).

## Training protocol

The pre-trained networks used ImageNet weights with a small classification head. In general, training used class weights, batch size 32 and augmentation by rotation up to 20 degrees, width and height shifts and horizontal flipping, with zoom and shear (usually 10%) added for most networks. Most networks used early stopping on validation loss with restoration of the best weights, and most also reduced the learning rate on plateau. The settings of each network are in its notebook. The exceptions are the four NIH networks below, which were trained for a fixed number of epochs without early stopping or learning-rate reduction.

| NIH network | Epochs | Checkpoint used |
|-------------|--------|-----------------|
| EfficientNetB4 | 15 | Epoch 15 |
| MobileNetV2 | 20 | See notebook |
| SE-MobileNetV2 | 20 | See notebook |
| ResNet50 | 10 with the base frozen, then 10 with fine-tuning | Final epoch |

Pixel values were divided by 255 for the stand-alone VGG16, ResNet50, DenseNet-121 and squeeze-and-excitation networks. The model-specific pre-processing function was used for Xception, the ensemble members, MobileNetV2 and EfficientNetB4. The networks shared the data, the loss family and the augmentation core but not identical training schedules, so differences between architectures should be read as indicative.

## Notes for readers

- **ResNet50 (NIH).** The saved file `Pneumo_ResNet50_Best_model.keras` holds the final epoch and not the best epoch. During the 10 frozen-base epochs the validation accuracy stayed constant, and learning took place during fine-tuning. The head flattens the final feature map, and the network has 49,278,337 parameters, of which 40,140,801 were trainable after fine-tuning.
- **EfficientNetB4 (NIH).** The validation loss was lowest after the first epoch and rose afterwards, so the model is overfitted to some degree. The epoch 15 checkpoint was used in all evaluations.
- **Ensemble evaluation.** The vote weights stored with the original ensemble had been tuned on the first 400 test images. The manuscript reports weights tuned on the validation images (96.63% accuracy), and the archived result (96.79%) is stated alongside.
- **Timing.** Inference was timed in a Colab session whose processor runs at 2.2 GHz with one physical core, and the times were multiplied by 1.47 to scale the clock to the Raspberry Pi 4 Model B. This does not reproduce the board. Floating-point TensorFlow Lite files were as fast as or faster than the quantised files on this host, so the times are estimates of relative cost, and latency on a physical board has not been measured.
- **Thread settings.** The environment variables for the number of threads were set after TensorFlow was imported and do not govern the TensorFlow Lite interpreter. The comparison with 1, 2 and 4 interpreter threads is in the verification notebook.
- **Grad-CAM.** The layer used is the last layer whose name contains `conv` and whose output is four-dimensional (`top_conv` for EfficientNetB4 and `block14_sepconv2_act` for Xception). Maps from the compiled implementation were identical to those from the standard implementation at the native input size. A 128 × 128 input changed the predictions.
- **Software.** The verification notebooks were run with TensorFlow 2.21 and Keras 3.13.

## Trained models

<!-- TO COMPLETE BEFORE SUBMISSION: state where the trained Keras and TensorFlow Lite files can be downloaded (for example a GitHub release or a Drive folder) and give their sizes. -->

## Licence and citation

The code is released under the licence in the repository root. Please cite the manuscript when it is published. Until then:

```bibtex
@unpublished{konadu2026efficient,
  title  = {Computationally Efficient {AI} Models for Pneumonia Detection in Low-Resource Settings},
  author = {Konadu, Evans and Abbasi, Negar and Thakur, Siddhesh and Rahim, Tariq and Bakas, Spyridon and Makris, Dimitrios},
  note   = {Manuscript prepared for submission to Discover Artificial Intelligence},
  year   = {2026}
}
```

The research models are not intended for clinical use.
