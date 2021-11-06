import os


def iterate_dir(dir_path):
    """
    returns all the names of the files in the directory path supplied as 
    argument to the function.
    """
    return [os.path.join(dir_path, f) for f in os.listdir(dir_path)]


def prepare_data(data_dir):
    """
    prepare training/testing data
    return a list of all the image files, along with a list of the classes the image
    belongs to
    """
    images = []
    classes = []

    # each data dir contains subdirs of images for each class
    for class_dir in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_dir)
        class_name = os.path.basename(class_path)
        for f in os.listdir(class_path):
            images.append(os.path.join(class_path, f))
            classes.append(class_name)

    return images, classes


if __name__ == '__main__':
    # testing
    images, data = prepare_data('data/train')
    print(images)
    print(data)