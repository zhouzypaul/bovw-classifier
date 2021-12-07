import pickle
import itertools

import cv2
import numpy as np
from sklearn import cluster, svm


class BOVWClassifier:
    """
    bag of visual words (BOVW) classifier
    used for image classification
    args:
        num_clusters: number of clusters to use for kmeans clustering
        num_sift_keypoints: number of SIFT keypoints to use for SIFT extraction,
                            if None, use cv2 will auto pick a number of keypoints
    """
    def __init__(self, num_clusters=50, num_sift_keypoints=None):
        self.num_clusters = num_clusters
        self.num_keypoints = num_sift_keypoints
        self.kmeans_cluster = None
        self.svm_classifier = None
        if num_sift_keypoints is not None:
            self.sift_detector = cv2.SIFT_create(nfeatures=num_sift_keypoints)
        else:
            self.sift_detector = cv2.SIFT_create()

    def fit(self, X=None, Y=None):
        """
        train the classifier in 3 steps
        1. train the kmeans classifier based on the SIFT features of images
        2. extract the histogram of the SIFT features
        3. train an SVM classifier based on the histogram features
        args:
            x: a list of images
            y: a list of labels (str)
        """
        X = list(X)  # in case X, Y are numpy arrays
        Y = list(Y)

        # get sift features
        sift_features = self.get_sift_features(images=X)
        # train kmeans
        self.train_kmeans(sift_features=sift_features)
        # get histogram
        hist_features = self.histogram_from_sift(sift_features=sift_features)
        # train svm
        self.svm_classifier = svm.SVC()
        self.svm_classifier.fit(hist_features, Y)

    def predict(self, X):
        """
        test the classifier
        args:
            X: a list of images
        """
        assert self.svm_classifier is not None
        X = list(X)  # in case X is numpy array

        # preprocess the images
        sift_features = self.get_sift_features(images=X)  # get sift features
        hist_features = self.histogram_from_sift(sift_features=sift_features)  # get histogram

        return self.svm_classifier.predict(hist_features)
    
    def __getstate__(self):
        """
        return state values that needs to be pickled

        this functione is called when pickling the class during self.save(), and
        is needed because cv2.SIFT is not picklable
        """
        return self.num_clusters, self.num_keypoints, self.kmeans_cluster, self.svm_classifier
    
    def __setstate__(self, state):
        """
        restore the state values after unpickling
        """
        self.num_clusters, self.num_keypoints, self.kmeans_cluster, self.svm_classifier = state
        if self.num_keypoints is not None:
            self.sift_detector = cv2.SIFT_create(nfeatures=self.num_keypoints)
        else:
            self.sift_detector = cv2.SIFT_create()
    
    def save(self, save_path):
        """
        save the classifier to disk
        """
        with open(save_path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, load_path):
        """
        init a classifier by loading it from disk
        """
        with open(load_path, 'rb') as f:
            return pickle.load(f)
    
    def get_sift_features(self, images):
        """
        extract the SIFT features of a list of images
        args:
            images: a list of RGB/GrayScale images
        return:
            a list of SIFT features
        """
        keypoints = self.sift_detector.detect(images)
        keypoints, descriptors = self.sift_detector.compute(images, keypoints)
        return descriptors  # type: tuple
    
    def train_kmeans(self, sift_features):
        """
        train the kmeans classifier using the SIFT features
        args:
            sift_features: a list of SIFT features
        """
        # reshape the data
        # each image has a different number of descriptors, we should gather 
        # them together to train the clustering
        sift_features=np.array(sift_features, dtype=object)
        sift_features=np.concatenate(sift_features, axis=0)

        # train the kmeans classifier
        if self.kmeans_cluster is None:
            self.kmeans_cluster = cluster.MiniBatchKMeans(n_clusters=self.num_clusters, random_state=0).fit(sift_features)
        else:
            self.kmeans_cluster.partial_fit(sift_features)
    
    def histogram_from_sift(self, sift_features):
        """
        transform the sift features by putting the sift features into self.num_clusters 
        bins in a histogram. The counting result is a new feature space, which
        is used directly by the SVM for classification
        args:
            sift_features: a list of SIFT features

        this code is optimized for speed, but it's doing essentially the following:

        hist_features = []
        for sift in sift_features:
            # classification of all descriptors in the model
            # sift.shape == (n_descriptors, 128)
            predicted_cluster = self.kmeans_cluster.predict(sift)  # (n_descriptors,)
            # calculates the histogram
            # hist, bin_edges = np.histogram(predicted_cluster, bins=self.num_clusters)  # (num_clusters,)
            hist = np.bincount(predicted_cluster, minlength=self.num_clusters)  # (num_clusters,)
            # histogram is the feature vector
            hist_features.append(hist)
        hist_features = np.asarray(hist_features)
        return hist_features
        """
        assert self.kmeans_cluster is not None, "kmeans classifier not trained"

        n_descriptors_per_image = [len(sift) for sift in sift_features]
        idx_num_descriptors = list(itertools.accumulate(n_descriptors_per_image))
        sift_features_of_all_images = np.concatenate(sift_features, axis=0)

        predicted_cluster_of_all_images = self.kmeans_cluster.predict(sift_features_of_all_images)  # (num_examples,)
        predicted_clusters = np.split(predicted_cluster_of_all_images, indices_or_sections=idx_num_descriptors)
        predicted_clusters.pop()  # remove the last element, which is empty due to np.split
        
        hist_features = np.array([np.bincount(predicted_cluster, minlength=self.num_clusters) for predicted_cluster in predicted_clusters])
        return hist_features
