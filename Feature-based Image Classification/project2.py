import numpy as np
import cv2
import os
from sklearn.cluster import KMeans
from sklearn.metrics import plot_confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from matplotlib import pyplot as plt


def generate_descriptors(path):
    descriptors = []
    for img in os.listdir(path):
        I = cv2.imread(f'{path}{img}')
        I= cv2.cvtColor(I,cv2.COLOR_BGR2GRAY) #change to gray
        sift = cv2.SIFT_create()
        [f,descriptor] = sift.detectAndCompute(I,None)
        descriptors.append(descriptor)
    return descriptors


def generate_histograms(path, descriptors):
    image_histogram_list = []
    for image_descriptor in descriptors:
        image_histogram = np.zeros(100)
        for feature in image_descriptor:
            feature = feature.reshape(1, 128)
            index = kmeans.predict(feature)
            image_histogram[index] += 1
        image_histogram_list.append(np.divide(image_histogram,sum(image_histogram))) #normalize
    return image_histogram_list


def generate_histogram_plot(image_histogram, clusters):
    plt.bar(np.arange(clusters), image_histogram)
    plt.xlabel("Visual Words")
    plt.ylabel("Count")
    plt.title("Histogram")
    plt.show()


def generate_labels(path):
    labels = []
    for img in os.listdir(path):
        labels.append(img[:3])
    return labels

# 3.1 Find SIFT features
train_path = './Project2_data/TrainingDataset/'
train_descriptors = generate_descriptors(train_path)

# 3.2 Clustering
print("Starting Kmeans...")
kmeans = KMeans(n_clusters = 100, max_iter = 100, n_init = 3).fit(np.concatenate(train_descriptors))
print("Done.")

# 3.3 Form Histograms
train_histograms = generate_histograms(train_path, train_descriptors)
generate_histogram_plot(train_histograms[2],100) #just an example

# 3.4 Prepare for Classification
test_path = './Project2_data/TestingDataset/'
test_descriptors = generate_descriptors(test_path)
test_histograms = generate_histograms(test_path, test_descriptors)

generate_histogram_plot(test_histograms[2],100) #just an example

train_labels = generate_labels(train_path)
test_labels = generate_labels(test_path)


# 3.5 Classification
neigh = KNeighborsClassifier(n_neighbors=1).fit(train_histograms, train_labels)
print(accuracy_score(test_labels, neigh.predict(test_histograms)))
plot_confusion_matrix(neigh, test_histograms, test_labels)  
plt.show() 

# 3.6 Linear Support Vector Machine
linearSVC = SVC(kernel="linear").fit(train_histograms, train_labels)
print(accuracy_score(test_labels, linearSVC.predict(test_histograms)))
plot_confusion_matrix(linearSVC, test_histograms, test_labels)  
plt.show() 

# 3.7 Kernel Support Vector Machine
rbfSVC = SVC(kernel="rbf").fit(train_histograms, train_labels)
print(accuracy_score(test_labels, rbfSVC.predict(test_histograms)))
plot_confusion_matrix(rbfSVC, test_histograms, test_labels)  
plt.show() 

