# CIFAR-10 CNN

A hands-on deep learning project exploring the CIFAR-10 image classification dataset and building a Convolutional Neural Network (CNN) with PyTorch.

## Current Progress

- Downloaded the Kaggle CIFAR-10 competition files using KaggleHub.
- Explored `trainLabels.csv` and confirmed 50,000 labeled training examples across 10 balanced classes.
- Explored image dimensions (32x32 RGB) and verified consistency.
- Visualized sample images and class distributions.
- Computed dataset-wide RGB mean and standard deviation for normalization.
- Created custom `Cifar10Dataset` with data augmentation (`RandomCrop`, `RandomHorizontalFlip`).
- Built a deep 3-block CNN architecture with Batch Normalization, Dropout, and standard Pooling.
- Implemented complete Training & Validation pipeline with Adam optimizer, `ReduceLROnPlateau`, and test prediction visualization.

## Dataset

The downloaded files are stored locally through KaggleHub and are not committed to this repository. The competition data includes `train.7z`, `test.7z`, `trainLabels.csv`, and `sampleSubmission.csv`.

- [x] Explore image files and dimensions
- [x] Visualize sample images and labels
- [x] Compute dataset normalization stats & build PyTorch Dataset
- [x] Build deep CNN architecture with Batch Normalization & Dropout
- [x] Implement training and validation loops with learning rate scheduling
- [ ] Run full multi-epoch training and log final evaluation metrics
- [ ] Generate predictions on Kaggle test set (`test.7z`) for submission

## Tools

- Python
- PyTorch
- Pandas
- NumPy
- Matplotlib
- KaggleHub
