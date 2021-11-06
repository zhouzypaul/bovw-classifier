import cv2
import numpy as np
from sklearn import cluster, svm


class BOVWClassifier:
    """
    bag of visual words (BOVW) classifier
    used for image classification
    args:
        num_clusters: number of clusters to use for kmeans clustering
    """
    def __init__(self, num_clusters=50):
        self.num_clusters = num_clusters
        self.kmeans_cluster = None
        self.svm_classifier = None

    def train(self, X=None, Y=None):
        """
        train the classifier in 3 steps
        1. train the kmeans classifier based on the SIFT features of images
        2. extract the histogram of the SIFT features
        3. train an SVM classifier based on the histogram features
        args:
            x: a list of images
            y: a list of labels (str)
        """
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

        # preprocess the images
        sift_features = self.get_sift_features(images=X)  # get sift features
        hist_features = self.histogram_from_sift(sift_features=sift_features)  # get histogram

        return self.svm_classifier.predict(hist_features)
    
    def save(self, save_path):
        """
        save the classifier to disk
        """
        # TODO
    
    def load(self, load_path):
        """
        init a classifier by loading it from disk
        """
        # TODO
    
    def get_sift_features(self, images):
        """
        extract the SIFT features of a list of images
        args:
            images: a list of RGB/GrayScale images
        return:
            a list of SIFT features
        """
        sift_features = []
        for image in images:
            # Convert them to grayscale
            image =cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # SIFT extraction
            sift = cv2.xfeatures2d.SIFT_create()
            kp, descriptors = sift.detectAndCompute(image, None)
            #append the descriptors to a list of descriptors
            sift_features.append(descriptors)  # each descriptor of shape (n_descriptors, 128)

        return sift_features
    
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
        self.kmeans_cluster = cluster.MiniBatchKMeans(n_clusters=self.num_clusters, random_state=0).fit(sift_features)
    
    def histogram_from_sift(self, sift_features):
        """
        transform the sift features by putting the sift features into self.num_clusters 
        bins in a histogram. The counting result is a new feature space, which
        is used directly by the SVM for classification
        args:
            sift_features: a list of SIFT features
        """
        assert self.kmeans_cluster is not None, "kmeans classifier not trained"

        hist_features = []
        for sift in sift_features:
            # classification of all descriptors in the model
            predicted_cluster = self.kmeans_cluster.predict(sift)
            # calculates the histogram
            hist, bin_edges = np.histogram(predicted_cluster, bins=self.num_clusters)
            # histogram is the feature vector
            hist_features.append(hist)

        hist_features = np.asarray(hist_features)
        return hist_features
