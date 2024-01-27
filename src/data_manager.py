import os 
import numpy as np
from skimage import io
from tqdm import tqdm


def get_image_paths(data_path):
    for (dir, dir_names, file_names) in os.walk(data_path):
        for file_name in file_names:
            image_file = os.path.join(dir, file_name)
            yield image_file


def get_images(image_files):

    images = []
    for image_file in tqdm(image_files):
        image = io.imread(image_file)
        images.append(image)

    return np.array(images)


def get_labels(image_files, dataset):

    if dataset == 'ck':

        emotion_dict = {
            'surprise': 'Surprise',
            'sadness': 'Sadness',
            'contempt': 'Neutral',
            'happy': 'Happiness',
            'fear': 'Fear',
            'disgust': 'Disgust',
            'anger': 'Anger'
        }

        labels = []
        for image_file in image_files:
            image_label = emotion_dict[image_file.split(os.path.sep)[-2]]
            labels.append(image_label)

    if dataset == 'jaffe':

        emotion_dict = {
            'SU': 'Surprise',
            'SA': 'Sadness',
            'NE': 'Neutral',
            'HA': 'Happiness',
            'FE': 'Fear',
            'DI': 'Disgust',
            'AN': 'Anger'
        }

        labels = []
        for image_file in image_files:
            image_label = emotion_dict[image_file.split(os.path.sep)[-1].split('.')[1][:2]]
            labels.append(image_label)

    return np.array(labels)
    