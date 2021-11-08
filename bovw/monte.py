from pathlib import Path

import cv2
import numpy as np
from pfrl.wrappers import atari_wrappers
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score

from bovw.utils import prepare_data
from bovw import BOVWClassifier


def get_player_position(ram):
    """
    given the ram state, get the position of the player
    """
    def _getIndex(address):
        assert type(address) == str and len(address) == 2
        row, col = tuple(address)
        row = int(row, 16) - 8
        col = int(col, 16)
        return row * 16 + col
    def getByte(ram, address):
        # Return the byte at the specified emulator RAM location
        idx = _getIndex(address)
        return ram[idx]
    # return the player position at a particular state
    x = int(getByte(ram, 'aa'))
    y = int(getByte(ram, 'ab'))
    return x, y


def distance(pos1, pos2):
    """
    l2 distance
    """
    return np.linalg.norm(np.array(pos1) - np.array(pos2))


def collect_monte_frames(total_steps=200, save_dir='data/monte'):
    """
    run a monte game using random actions, and collect frames from it
    save the resulting frames into two directories: start, non_start
    """
    # make env
    env = atari_wrappers.wrap_deepmind(
        atari_wrappers.make_atari('MontezumaRevengeNoFrameskip-v4', max_frames=30*60*60),
        episode_life=True,
        clip_rewards=True,
    )
    env.seed(0)

    # collection
    state = env.reset()
    step = 0
    while step < total_steps:

        # save image
        im = np.array(state)[-1, :, :]
        pos = get_player_position(env.unwrapped.ale.getRAM())
        if distance(pos, (77, 235)) < 1:
            file_dir = Path(save_dir) / 'start'
        else:
            file_dir = Path(save_dir) / 'non_start'
        file_dir.mkdir(exist_ok=True, parents=True)
        plt.imsave(file_dir.joinpath(f"step_{step}.png"), im)

        # control
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        if done or info.get('needs_reset', False):
            state = env.reset()
        step += 1


def visualize_sift_features(images, save_dir='results/monte_sift'):
    """
    visualize the sift features of the frames
    """
    # prepare dir 
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    for i, image in enumerate(images):
        # turn to gray scale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # SIFT extraction
        sift = cv2.xfeatures2d.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(image, None)

        # draw the detected key points
        sift_image = cv2.drawKeypoints(gray_image, keypoints, image)
        # save the image
        cv2.imwrite(str(save_dir.joinpath(f"{i}_sift.png")), sift_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # collect frames
    collect_monte_frames(total_steps=200, save_dir='data/monte_train')
    collect_monte_frames(total_steps=30 ,save_dir='data/monte_test')

    # visualize sift features
    image_paths, classes = prepare_data('data/monte_train')
    images = [cv2.imread(path, cv2.IMREAD_COLOR) for path in image_paths]
    visualize_sift_features(images, save_dir='results/monte_sift')

    # load data
    training_image_paths, training_classes = prepare_data('data/monte_train')
    training_images = [cv2.imread(path, cv2.IMREAD_COLOR) for path in training_image_paths]
    testing_image_paths, testing_classes = prepare_data('data/monte_test')
    testing_images = [cv2.imread(path, cv2.IMREAD_COLOR) for path in testing_image_paths]

    # train the classifier
    classifier = BOVWClassifier(num_clusters=50)
    classifier.train(training_images, training_classes)

    # test the classifier
    predictions = classifier.predict(testing_images)
    accuracy = accuracy_score(testing_classes, predictions)
    print("accuracy: {}".format(accuracy))

    
