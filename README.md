# Kernel-PCA-Facial-Recognition

### Abstract

PCA is a classical method based on statistical features. Applied in facial recognition, this method demonstrated its effectiveness by the correct recognition results.

One of the significant issues is the presence of _nonlinear_ correlations among image pixels. PCA performs only on _linear_ space and is not efficient when capturing information about the nonlinear one. In addressing this problem, Kernel Principal Component Analysis (KPCA) proposes reliance on kernel space theory as a nonlinear extension of PCA.

### Description 

* This project presents the use of the Kernel Principal Component Analysis method for facial recognition.  

* Using machine learning, the method trains a data set (images of faces as pixel matrices). It picks a sample test image and finds the closest similar matrix of pixels from the set: each line represents the pixel vector of an image 'i'. The data file contains 40 persons with 10 images each (400-pixel vectors).
* As the first step, the `KPCA` function calculates the kernel matrix, centering it in the nonlinear feature space. It calculates the eigenvectors and eigenvalues of this K matrix, projecting them in the PCA subspace. The data is ready to be used by a trained "SVM" classifier (`SVC` object), which predicts the closest image to the initial test sample chosen.

### Directory Files 

* `KPCA_iterative.py` file: The iterative KPCA method.

* `KPCA_predef.py` file: The predefined KPCA method, using the scikit-learn library.

* `face_data.csv` file: The data set.
  
### Getting Started

#### Requirements

* Python IDE
* Git

#### Setup

1. Clone the repository and install the dependencies.
 
```bash
git clone https://github.com/gmelania/Kernel-PCA-Facial-Recognition.git
```

2. Open the "KPCA Facial Recognition" in your IDE, and access the `KPCA_iterative.py` file. Run the code using the terminal command:
   
```bash
python KPCA_iterative.py
```

3. The algorithm will pick a random image and search for it in the data set, comparing the result to the original.

