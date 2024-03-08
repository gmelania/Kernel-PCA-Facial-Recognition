from sklearn.decomposition import KernelPCA
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import warnings


def plot_two_faces(index1, index2, data, image_size=(64, 64)):
    """
    Plotează două imagini ale fețelor la indexurile specificate din 
    setul de date pe același grafic.

    :param index1: Indexul primei imagini de plottat.
    :param index2: Indexul celei de-a doua imagini de plottat.
    :param data: DataFrame-ul care conține datele imaginii.
    :param image_size: Dimensiunea la care imaginea trebuie redată (64x64).
    """
    if 'target' in data.columns:
        # Excludem coloana 'target' dacă există
        image_data1 = data.drop('target', axis=1).iloc[index1]
        image_data2 = data.drop('target', axis=1).iloc[index2]
    else:
        image_data1 = data.iloc[index1]
        image_data2 = data.iloc[index2]

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].imshow(np.array(image_data1).reshape(image_size), cmap='gray')
    axes[0].set_title(f"Image at index {index1}")
    axes[0].axis('off')

    axes[1].imshow(np.array(image_data2).reshape(image_size), cmap='gray')
    axes[1].set_title(f"Image at index {index2}")
    axes[1].axis('off')

    plt.show()


warnings.filterwarnings('ignore')

df = pd.read_csv('face_data.csv')

df['target'].nunique()
print('df:', df)


# previzualizarea datelor
# def plot_faces(pixels, title):
#     fig, axes = plt.subplots(1, len(pixels),
#                              figsize=(10, 5))
#     for i, ax in enumerate(axes):
#         ax.imshow(np.array(pixels[i])
#                   .reshape(64, 64), cmap='gray')
#         ax.set_title(title[i])
#     plt.show()


def kernel_pca(X, n_components=107, gamma=0.01):

    # 1. Calculam matricea de kernel

    # gasim distantele euclidiene dintre date
    from scipy.spatial.distance import cdist
    pairwise_sq_dists = cdist(X, X, 'sqeuclidean')

    # aplicam kernelul RBF asupra distantelor => se obtine o matrice de similaritate bazata pe RBF
    K = np.exp(-gamma * pairwise_sq_dists)

    # 2. Centram datele

    n_samples = X.shape[0]
    # cream matricea de medii
    one_n = np.ones((n_samples, n_samples)) / n_samples
    # se elimina componenta medie pe coloane și linii apoi se adauga inapoi media totala matricei
    K_centered = K - one_n @ K - K @ one_n + one_n @ K @ one_n

    # 3. Calculam vectorii si valorile proprii ale matricei K

    eigvals, eigvecs = np.linalg.eigh(K_centered)

    # vectorii proprii ai celor mai mari n_components=107 valori proprii
    alphas = eigvecs[:, -n_components:]
    # valorile proprii corespunzatoare vectorilor proprii selectati
    lambdas = eigvals[-n_components:]
    # calculam cumulative percentage al valorilor proprii
    cumulative_percentage = np.cumsum(lambdas) / np.sum(lambdas) * 100

    # 4. Proiectam datele in subspatiul PCA

    X_pc = alphas * np.sqrt(lambdas)

    return X_pc, cumulative_percentage


X = df.drop('target', axis=1)
Y = df['target']

# Obtinem datele transformate in spatiul de dimensiune redusa
X_train_kpca, cum_per = kernel_pca(
    X.values, n_components=107, gamma=0.01)

# Antrenam clasificatorul SVM pe datele transformate
classifier = SVC(C=1.0, kernel='linear')
classifier.fit(X_train_kpca, Y)

# Proiectam imaginea testata pe subspatiul creat

lower_limit = 0
upper_limit = 399
random_index = random.randint(lower_limit, upper_limit)

# Se aplica KPCA pe linia sample de la indicele random selectat
X_test_kpca = X_train_kpca

# transformam imaginea sample
X_test_sample_transformed = X_test_kpca[random_index]
X_test_sample_transformed = X_test_sample_transformed.reshape(1, -1)

# se prezice eticheta cu imaginea de test transformata
predicted_label = classifier.predict(X_test_sample_transformed)

# accesam eticheta unde valoarea este egala cu eticheta prezisa
found_index = np.where(Y.values == predicted_label[0])[0][0]
print('Tested person was labeled: ', random_index)
print('Found person was labeled: ', found_index)


plot_two_faces(random_index, found_index, df)

# Plot cumulative percentage al tuturor eigenvalues
plt.figure(figsize=(8, 6))
plt.plot(cum_per, marker='o', linestyle='-')
plt.title('Cumulative Percentage of Total Eigenvalues')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Percentage')
plt.grid(True)
plt.savefig('cumulative_percentage.png')

# afisare grafuri

kpca = KernelPCA(n_components=107, kernel='rbf', gamma=0.01)
X_train_kpca2 = kpca.fit_transform(X)

plt.figure(figsize=(10, 7))

# Afișăm rezultatele metodei kernel iterative
plt.plot(cum_per, label='Kernel Iterative')
