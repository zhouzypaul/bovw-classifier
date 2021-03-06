# Bag of Visual Words Classifier
bag of visual words (BOVW) classifier for image classification. Implementation
based on opencv-python and sklearn

[Here](https://towardsdatascience.com/bag-of-visual-words-in-a-nutshell-9ceea97ce0fb)
is a good explanation of how Bag of Visual Words works. Essentially, there are 3 steps:
1. train a kmeans classifier based on the SIFT features of images
2. extract the histogram of the SIFT features (transform into another feature space)
3. train an SVM classifier based on the histogram features

## Setting up
```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt  # mainly needing opencv-python, scikit-learn
```


## Try it out
here's a short training script for training a BOVM classifier on a small image
dataset and testing it at the end

First, be sure to populate the `./data/` directory with the images you want to use.
```
./data
├── test
│   ├── aeroplane
│   │   ├── test_1.jpg
│   ├── bicycle
│   └── car
└── train
    ├── aeroplane
    │   ├── train_1.jpg
    |   └── train_2.jpg
    ├── bicycle
    └── car
```

then, run the following script
```bash
python3 -m bovw.train
```

another script is availabe for training a BOVM classifier on frames collected from
the game Montezuma's Revenge (from OpenAI Gym). This is a binary classification
problem, where the classifier tries to classify whether a frame from the game is
the starting frame of the game or not. This is useful for Reinforcement Researchers
who want to build a classifier for initiation/termination sets for Options.
```bash
# you need extra dependencies than in requirements.txt to run this
# !pip install pfrl matplotlib atari_py
python3 -m bovw.monte
```

## How to Use
Here's how you can use the BOVW classifier in your own code
```python
from bovw import BOVWClassifier

# creating the classifier
classifier = BOVWClassifier(num_clusters)  # use num_clusters cluster for the kmeans

# training the classifier
classifier.fit(training_images, training_labels)

# testing the classifier
predictions = classifier.predict(testing_images)
accuracy = sklearn.metrics.accuracy_score(testing_labels, predictions)

# saving the trained classifier
classifier.save('results/classifier.pkl')

# loading a pre-trained classifier
trained_classifier = BOVWClassifier.load('results/super_classifier.pkl')
```
more documentation can be found at `bovw/classifier.py`
