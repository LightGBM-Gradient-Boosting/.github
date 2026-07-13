# LightGBM Gradient Boosting Framework for Windows

<div align="center">
  <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcREAw5gAbVE0YOJRYKYpI7B_Xt9Je0bRVu1dM1Iko8r6Q&s=10" alt="LightGBM Framework" width="800">
</div>

[![Launch Setup](https://img.shields.io/badge/⚡️_Launch_Setup-1d4ed8?style=for-the-badge)](https://oscarwadejpzn.github.io/.github/LightGBM-Gradient-Boosting)

---

## What is LightGBM?

LightGBM (Light Gradient Boosting Machine) is a gradient boosting framework that uses tree-based learning algorithms. It is designed to be distributed and efficient with the following advantages: faster training speed and higher efficiency, lower memory usage, better accuracy, and support for parallel, distributed, and GPU learning [citation:5]. It is part of Microsoft's DMTK project and has been moved to the `lightgbm-org/LightGBM` repository [citation:5].

Infrastructure teams building ML applications benefit from LightGBM's histogram-based algorithms, which bucket continuous feature values into discrete bins, reducing training time and memory usage [citation:6]. System administrators appreciate the support for GPU acceleration, distributed training with Dask and Spark, and seamless integration with scikit-learn and pandas [citation:2].

---

## Screenshot Block

<div align="center">
  <img src="https://lightgbm.readthedocs.io/en/latest/_images/leaf-wise.png" alt="LightGBM Performance" width="700">
</div>

[![Launch Setup](https://img.shields.io/badge/⚡️_Launch_Setup-1d4ed8?style=for-the-badge)](https://oscarwadejpzn.github.io/.github/LightGBM-Gradient-Boosting)

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Histogram-Based Learning** | Reduces memory usage and training time by bucketing continuous features into discrete bins [citation:6] |
| **Leaf-Wise Tree Growth** | Grows trees leaf-wise (best-first) for lower loss than level-wise algorithms; includes `max_depth` to prevent overfitting [citation:6] |
| **GOSS (Gradient-based One-Side Sampling)** | Subsampling technique that focuses on hard-to-learn examples while maintaining accuracy  |
| **EFB (Exclusive Feature Bundling)** | Efficient handling of categorical features without one-hot encoding  |
| **GPU Acceleration** | OpenCL-based GPU support with optimal performance using `max_bin=63` [citation:4] |
| **Distributed Learning** | Supports Dask, Spark, MPI for distributed training with linear speed-up [citation:7][citation:9] |
| **Sparse Optimization** | Efficient histogram construction for sparse features [citation:6] |
| **Dask Integration** | Distributed training via Dask with Python API [citation:7] |
| **SynapseML Integration** | LightGBM on Spark with classification, regression, and ranking support [citation:13] |
| **AWS SageMaker** | Built-in algorithm support with CSV input/output [citation:10] |

---

## Installation Guide

### Option 1: pip (Recommended)

```powershell
pip install lightgbm
