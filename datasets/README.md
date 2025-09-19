# **Dataset Selection and Rationale**

This repository uses two publicly available chest X-ray datasets to train and evaluate deep learning models for pneumonia detection. Each dataset offers distinct characteristics in terms of patient demographics, class distribution, and imaging modality.

## NIH ChestX-ray14 Dataset

A large-scale dataset comprising **112,120** frontal-view chest X-ray images from **30,805** unique patients, developed by Wang et al. (2017). It includes 14 thoracic disease labels, with multi-label annotations per image.

- **Challenge**: Severe under-representation of pneumonia cases, with a class imbalance ratio of approximately **1:42**.

**Citation**:  
Wang X, Peng Y, Lu L, Lu Z, Bagheri M, Summers RM.  
*ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases*.  
IEEE CVPR 2017.

**Dataset Source**:  
[NIH ChestX-ray14 Original Files and Documentation](https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938765345)

---

## Mendeley Chest X-ray Dataset

A paediatric dataset containing **5,856** chest X-ray images curated by Kermany et al. (2018), sourced from Guangzhou Women and Children's Medical Centre.

- **Composition**:  
  - **4,672** pneumonia cases  
  - **1,184** normal cases  
- **Advantage**: High pneumonia prevalence (**79.8%**) and paediatric-specific imaging characteristics.
- **Download Source**: Directly from Mendeley Data repository (Version 3).

**Citation**:  
Kermany, Daniel; Zhang, Kang; Goldbaum, Michael (2018).  
*Large Dataset of Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images*, Mendeley Data, V3.  
DOI: [10.17632/rscbjbr9sj.3](https://doi.org/10.17632/rscbjbr9sj.3)

**Dataset Link**:  
[Mendeley Chest X-ray Dataset on Mendeley Data](https://data.mendeley.com/datasets/rscbjbr9sj/3)



