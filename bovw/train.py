import cv2
import numpy as np
from sklearn import cluster
from sklearn import svm
from sklearn.metrics import accuracy_score

from bovw.utils import prepare_data


def read_and_clusterize(image_paths, num_cluster):
    """
    get SIFT descriptors from training images and train a k-means classifier    
    """
    sift_keypoints = []

    for path in image_paths:
        print(path)
        #read image
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        # Convert them to grayscale
        image =cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # SIFT extraction
        sift = cv2.xfeatures2d.SIFT_create()
        kp, descriptors = sift.detectAndCompute(image, None)
        #append the descriptors to a list of descriptors
        sift_keypoints.append(descriptors)

    sift_keypoints=np.asarray(sift_keypoints)
    sift_keypoints=np.concatenate(sift_keypoints, axis=0)

    #with the descriptors detected, lets clusterize them
    print("Training kmeans")    
    kmeans = cluster.MiniBatchKMeans(n_clusters=num_cluster, random_state=0).fit(sift_keypoints)
    #return the learned model
    return kmeans


def calculate_centroids_histogram(image_paths, model, num_clusters):
    """
    with the k-means model found, this code generates the feature vectors 
    by building an histogram of classified keypoints in the kmeans classifier 
    """
    feature_vectors=[]

    for path in image_paths:
        print(path)
        #read image
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        #Convert them to grayscale
        image =cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #SIFT extraction
        sift = cv2.xfeatures2d.SIFT_create()
        kp, descriptors = sift.detectAndCompute(image, None)
        #classification of all descriptors in the model
        predict_kmeans=model.predict(descriptors)
        #calculates the histogram
        hist, bin_edges=np.histogram(predict_kmeans, bins=num_clusters)
        #histogram is the feature vector
        feature_vectors.append(hist)

    feature_vectors=np.asarray(feature_vectors)
    return feature_vectors


def main(train_data='data/train', test_data='data/test', num_clusters=50):
    print("Step 0: prepare data")
    training_images, training_classes = prepare_data(train_data)
    testing_images, testing_classes = prepare_data(test_data)

    print("Step 1: Calculating Kmeans classifier")
    kmeans_model = read_and_clusterize(training_images, num_clusters)

    print("Step 2: Extracting histograms of training and testing images")
    train_featvec = calculate_centroids_histogram(training_images, kmeans_model, num_clusters)
    test_featvec = calculate_centroids_histogram(testing_images, kmeans_model, num_clusters)

    print("Step 3: Training the SVM classifier")
    classifier = svm.SVC()
    classifier.fit(train_featvec, training_classes)

    print("Step 4: Testing the SVM classifier")  
    predictions = classifier.predict(test_featvec)

    score=accuracy_score(testing_classes, predictions)
    print("Accuracy:" +str(score))


if __name__ == "__main__":
    main()