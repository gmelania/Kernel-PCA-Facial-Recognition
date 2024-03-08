import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.decomposition import KernelPCA
import warnings


def plot_face_at_index(index, data, image_size=(64, 64)):
    """
    Plotează imaginea feței la un index specificat din setul de date.

    :param index: Indexul imaginii de plottat.
    :param data: DataFrame-ul care conține datele imaginii.
    :param image_size: Dimensiunea la care imaginea trebuie redată (implicit 64x64).
    """
    if 'target' in data.columns:
        # excludem coloana 'target' dacă există
        image_data = data.drop('target', axis=1).iloc[index]
    else:
        image_data = data.iloc[index]

    plt.imshow(np.array(image_data).reshape(image_size), cmap='gray')
    plt.title(f"Image at index {index}")
    plt.axis('off')  # ascunde axele
    plt.show()


def plot_two_faces(index1, index2, data, image_size=(64, 64)):
    """
    Plotează două imagini ale fețelor la indexurile specificate din setul de date pe același grafic.

    :param index1: Indexul primei imagini de plottat.
    :param index2: Indexul celei de-a doua imagini de plottat.
    :param data: DataFrame-ul care conține datele imaginii.
    :param image_size: Dimensiunea la care imaginea trebuie redată (implicit 64x64).
    """
    if 'target' in data.columns:
        # excludem coloana 'target' dacă există
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
print('df[target]:', df['target'])
print('df[target].numuniq():', df['target'].nunique())

# previzualizarea datelor


def plot_faces(pixels, title):
    fig, axes = plt.subplots(1, len(pixels), figsize=(10, 5))
    for i, ax in enumerate(axes):
        ax.imshow(np.array(pixels[i]).reshape(64, 64), cmap='gray')
        ax.set_title(title[i])
    plt.show()


X = df.drop('target', axis=1)
Y = df['target']

# folosim functia predefinita Kernel PCA
kpca = KernelPCA(n_components=107, kernel='rbf', gamma=0.01)
X_train_kpca = kpca.fit_transform(X)

# antrenm calsificatorul SVM pe datele transformate
classifier = SVC(C=1.0, kernel='linear')
classifier.fit(X_train_kpca, Y)

# proiectam test image in subspatiul creat

lower_limit = 0
upper_limit = 399

random_index = random.randint(lower_limit, upper_limit)
X_test_sample = X.iloc[random_index, :]
X_test_kpca = kpca.transform([X_test_sample])

predicted_label = classifier.predict(X_test_kpca)

# accesam eticheta unde valoarea == eticheta prezisa
found_index = np.where(Y.values == predicted_label[0])[0][0]
print('Tested person was labeled: ', random_index)
print('Found person was labeled: ', found_index)


plot_two_faces(random_index, found_index, df)

component_indices = range(1, len(X_train_kpca[found_index]) + 1)
component_values = X_train_kpca[found_index]

# matricea kernel
kernel_matrix = np.dot(X_train_kpca, X_train_kpca.T)

# eigenvalues din matricea kernel
eigenvalues = np.abs(np.linalg.eigvals(kernel_matrix))


# calculam si plotam cumulative percentage al tuturor eigenvalues
cumulative_percentage = np.cumsum(eigenvalues) / np.sum(eigenvalues) * 100

plt.plot(cumulative_percentage, marker='o')
plt.title('Cumulative Percentage of Total Eigenvalues')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Percentage')
plt.grid(True)
plt.savefig('cumulative_percentage_kpca_predefined.png')
