# Computationally Efficient AI Models for Pneumonia Detection in Low-Resource Settings

Code accompanying the manuscript prepared for submission to *Discover Artificial Intelligence* (topical collection "Intelligent Imaging Technologies, Methods and Applications").

**Authors:** Evans Konadu, Negar Abbasi, Siddhesh Thakur, Tariq Rahim, Spyridon Bakas and Dimitrios Makris (corresponding author, Kingston University).

This folder holds the notebooks added for the journal manuscript. The training notebooks for the ten architectures and the Nuclear Ensemble, on both datasets, were written for the MSc dissertation and remain in the repository root under [`notebooks/`](../notebooks/) ([Mendeley](../notebooks/mendeley_dataset/) and [NIH](../notebooks/nih_dataset/)). The root [README](../README.md) describes the dissertation. The TensorFlow Lite conversion of the quantised models, and the original timing runs, are in the root notebooks `EfficientNetB4_NIH` and `Nuclear_Ensemble_Classifier_Mendeley`.

## Contents of this folder

| File | Content |
|------|---------|
| `Multi-Label_Pneumonia_Classification_EfficientNetB4.ipynb` | The 14-label EfficientNetB4 with multi-scale attention and a distribution-balanced loss: training on the over-sampled table, dynamic-range quantisation, the main evaluation of the original and quantised models (section headed MAIN EVALUATION), and timing. The cells before the main evaluation include the Youden-threshold analysis and trials with test-time augmentation and fine-tuning that are not reported in the manuscript |
| `Optimisation_NIH_Mendeley.ipynb` | Knowledge distillation into MobileNetV2 students (NIH EfficientNetB4, Mendeley Xception and the Nuclear Ensemble, with grid searches over the distillation settings), and re-evaluation of the Nuclear Ensemble with vote weights tuned on the validation images, for the original and INT8 base models |
| `Verification_and_Sensitivity_Analyses.ipynb` | Checks run after the main experiments, without retraining: patient overlap in the NIH binary folders, 95% intervals for the selected models, multi-label partition counts, FP32 TensorFlow Lite baselines for the NIH EfficientNetB4 and Xception, a paired comparison of Xception and its INT8 version, two Grad-CAM benchmarks, the ensemble input pipelines, a test-folder identity check, and an inventory of the training settings of the notebooks |
| `Supplementary_Analyses_DiscoverAI.ipynb` | Saved-file size audit, evaluation of the NIH distilled student, FP32 TensorFlow Lite baselines for the 14-label model and the ensemble, the Grad-CAM benchmark with its figures, the Grad-CAM target layers and the 128 × 128 input, timing at 1, 2 and 4 interpreter threads, timing with and without the XNNPACK delegate, parameter and multiply-accumulate counts, and the model optimisation figure |

Each section of the supplementary notebook is a self-contained cell that reads its inputs from Google Drive by path.

## Where each part of the manuscript comes from

| Manuscript item | Source |
|-----------------|--------|
| Tables 4 and 5 (single networks, ensemble members and ensemble) | Notebooks in [`notebooks/mendeley_dataset`](../notebooks/mendeley_dataset/) and [`notebooks/nih_dataset`](../notebooks/nih_dataset/) |
| Sizes in Tables 4 to 6 | Supplementary notebook, section 1 |
| Table 6 (quantisation and distillation), Appendix Table A4 | Quantisation: root notebooks `EfficientNetB4_NIH` and `Nuclear_Ensemble_Classifier_Mendeley`. Distillation and the validation-tuned ensemble: `Optimisation_NIH_Mendeley.ipynb`. NIH student on the current test folder: Supplementary, section 2 |
| Fig. 3 | Supplementary notebook, section 8 |
| Table 7 (inference time) | Original timing runs in the root notebooks. FP32 baselines: Supplementary, sections 3 and 4 (14-label model and ensemble), and Verification, section 4 (NIH EfficientNetB4 and Xception) |
| Appendix Tables A5 and A6 (threads, XNNPACK) | Supplementary notebook, timing cells |
| Table 8 (parameters and MACs) | Supplementary notebook, last two cells |
| Table 9 and Fig. 5 (Grad-CAM) | Supplementary notebook, sections 5 to 7 and the target-layer and 128 × 128 cells |
| Appendix Table A1 (NIH EfficientNetB4 files) | Root notebook `EfficientNetB4_NIH`; test-folder check in the Verification notebook |
| Fig. 4, Appendix Tables A2 and A3 (multi-label) | `Multi-Label_Pneumonia_Classification_EfficientNetB4.ipynb`; partition counts in Verification, section 3 |

## Data

- NIH ChestX-ray14: <https://nihcc.app.box.com/v/ChestXray-NIHCC> (accessed 17 January 2026)
- Mendeley chest X-ray dataset (Kermany et al.): <https://doi.org/10.17632/rscbjbr9sj.3>

The notebooks were run in Google Colab and read the images from Google Drive paths (for example `/content/drive/MyDrive/chest_xray` and `/content/drive/MyDrive/NIH_processed`). Change these paths to the location of the downloaded data before running. The multi-label notebook reads the over-sampled training table (`train_balanced_all.csv`) and the held-out file (`val_original.xlsx`) from `/content/drive/MyDrive/Multi_Label_Classification`.

## Dataset partitions

- **Mendeley:** the distributed partition, with 5,232 training and 624 test images. A random 10% of the training folder (523 images) was used for validation for the stand-alone networks and 20% (1,045 images) for the ensemble members and distillation students.
- **NIH binary task:** pneumonia and normal images were split at random at image level (seed 42), giving 1,784 test images (1,272 normal, 512 pneumonia). The training folder holds 5,320 images, including 332 augmented pneumonia images, and 20% of it was used for validation. Because the split is at image level, patients can appear in both folders, and section 1 of the Verification notebook examines this on the folders as they now exist (they were rebuilt after the models were trained, so the section is an estimate). Absolute figures for this task are therefore best read as paired comparisons between an original model and its optimised versions, which are evaluated on the same images.
- **NIH multi-label task:** a patient-level partition (89,826 training images from 24,644 patients; 22,294 held-out images from 6,161 patients, divided into validation and test halves of 11,147 images). The two halves share 2,089 patients.

## Training protocol

The pre-trained networks used ImageNet weights with a small classification head. In general, training used class weights, batch size 32 and augmentation by rotation up to 20 degrees, width and height shifts and horizontal flipping, with zoom and shear (usually 10%) added for most networks. Most networks used early stopping on validation loss with restoration of the best weights, and most also reduced the learning rate on plateau. The settings of each network are in its notebook, and the last section of the Verification notebook lists them for every notebook found in the Google Drive used for the work. The exceptions are the four NIH networks below, which were trained for a fixed number of epochs without early stopping or learning-rate reduction.

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
- **Ensemble evaluation.** The vote weights stored with the original ensemble had been tuned on the first 400 test images. The manuscript reports weights tuned on the validation images (96.63% accuracy) and states the result with the original weights (96.79%) alongside. Both are reproduced in `Optimisation_NIH_Mendeley.ipynb` and in the Verification notebook.
- **Timing.** Inference was timed in a Colab session whose processor runs at 2.2 GHz with one physical core, and the times were multiplied by 1.47 to scale the clock to the Raspberry Pi 4 Model B. This does not reproduce the board. Floating-point TensorFlow Lite files were as fast as or faster than the quantised files on this host, so the times are estimates of relative cost, and latency on a physical board has not been measured.
- **Thread settings.** The environment variables for the number of threads do not govern the TensorFlow Lite interpreter, whose thread count is set by its own argument. The comparison with 1, 2 and 4 interpreter threads is in the Supplementary notebook.
- **Grad-CAM.** Table 9 of the manuscript comes from the Supplementary notebook, which uses the last layer whose name contains `conv` and whose output is four-dimensional (`top_conv` for EfficientNetB4 and `block14_sepconv2_act` for Xception). The two Grad-CAM cells in the Verification notebook are additional runs that use the last four-dimensional output (`top_activation` for EfficientNetB4), and their timings differ slightly. In all runs the compiled implementation gave the same maps as the standard one at the native input size, and a 128 × 128 input changed the predictions.
- **Confidence intervals.** The 95% intervals in the Verification notebook treat images as independent and do not allow for several images per patient, so they are likely to be narrower than the true uncertainty.
- **Software.** The verification and supplementary notebooks were run with TensorFlow 2.21 and Keras 3.13.

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
