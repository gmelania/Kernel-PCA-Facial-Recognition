# Kernel PCA Facial Recognition Project
### Project Overview
This project presents the use of the Kernel Principal Component Analysis method for facial recognition. The method trains a data set (images of faces representing pixel matrices) to find the closest image to the sample test image picked. 
In the set, each line represents the pixel vector of an image 'i'. The data file contains 10 images for each person and 40 persons, so 400-pixel vectors. The algorithm picks a test image from the data set. As the first step, it calculates the kernel matrix, centering it in the feature space. Next, it calculates the eigenvectors and eigenvalues of the K matrix, projecting them in the PCA subspace. The data is ready for use by a trained SVM classifier ('SVC' object), which predicts the closest image label to the initial test image chosen.

The "KPCA_iterative.py" file: contains the iterative implemented KPCA method.

The "KPCA_predef.py" file: contains the predefined KPCA method, from the scikit-learn library.

The "face_data.csv" file: contains the data set.
### Getting Started
1. Download and unzip the directory.
2. Open the "KPCA_iterative.py" file and run the code using the terminal command: 
```console
python KPCA_iterative.py
```
3. The algorithm will pick a random image and search for it in the data set.
### Contributors
Ghirda Melania - Project Developer
