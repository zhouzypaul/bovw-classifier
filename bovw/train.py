import cv2
from sklearn.metrics import accuracy_score

from bovw.utils import prepare_data
from bovw import BOVWClassifier


def main(train_data='data/train', test_data='data/test', num_clusters=50):
    print("prepare data")
    training_image_paths, training_classes = prepare_data(train_data)
    training_images = [cv2.imread(path, cv2.IMREAD_COLOR) for path in training_image_paths]
    testing_image_paths, testing_classes = prepare_data(test_data)
    testing_images = [cv2.imread(path, cv2.IMREAD_COLOR) for path in testing_image_paths]

    print("train classifier")
    classifier = BOVWClassifier(num_clusters)
    classifier.train(training_images, training_classes)

    print("test classifier")
    predictions = classifier.predict(testing_images)
    accuracy = accuracy_score(testing_classes, predictions)
    print("accuracy: {}".format(accuracy))

    print("save classifier")
    classifier.save('results/classifier.pkl')

    print("loading classifier and testing again")
    duplicate_classifier = BOVWClassifier.load('results/classifier.pkl')
    predictions = duplicate_classifier.predict(testing_images)
    accuracy = accuracy_score(testing_classes, predictions)
    print("loaded classifier accuracy: {}".format(accuracy))


if __name__ == "__main__":
    main()