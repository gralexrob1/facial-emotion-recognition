# import sys
# sys.path.append('../src')

from data_manager import get_image_paths, get_images, get_labels
from score import get_score, print_score
from hog import compute_gradients, compute_histograms, block_normalization, compute_hog
from lbp import extract_lbp_features
from gabor import gabor_filter, gabor_filtering, gabor_process

import numpy as np
from tqdm import tqdm

from skimage import feature, transform

from sklearn.base import (
    BaseEstimator, 
    TransformerMixin
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix
)
from sklearn.svm import SVC


### PREPROCESSING ###


def preprocessing_pipeline(data_path, dataset, verbose=False):

    image_files = list(get_image_paths(data_path))

    images = get_images(image_files)
    labels = get_labels(image_files, dataset)

    assert images.shape[0] == labels.shape[0]

    if verbose:
        print(f"Dataset: {dataset}")
        print(f"Dataset length: {len(image_files)}")
        print(f"Image shape: {images[0].shape}")
    
    return images, labels



### CLASSIFICATION ###

def classifier_pipeline(
    features, labels, 
    test_size=0.3, random_state= 42, shuffle=True, 
    verbose=False
):

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels,
        test_size=test_size, random_state=random_state, shuffle=shuffle
    )

    svm_clf = SVC(
        kernel='rbf',
        gamma='scale',
        C= 10 #strength of the regularization is inversely proportional to C
    )
    
    svm_clf.fit(X_train, y_train)

    pred_train, metrics_train = get_score(svm_clf, X_train, y_train, train=True)
    pred_test, metrics_test = get_score(svm_clf, X_test, y_test, train=False)

    if verbose:
        print_score(y_train, pred_train, train=True)
        print_score(y_test, pred_test, train=False)

    return metrics_train, metrics_test



### LBP ###

# def lbp_pipeline(
#     images, 
#     P,
#     R,
#     resize=None,
#     verbose=False
# ):

#     features = []

#     for image in tqdm(images):

#         if resize is not None:
#             image = transform.resize(image, resize)

#         lbp_image = feature.local_binary_pattern(
#             image,
#             P=P,
#             R=R
#         )
#         image_features = lbp_image.flatten()
#         features.append(image_features)

#     features = np.array(features)
    
#     if verbose:
#         print(f"P: {P}")
#         print(f"R: {R}")
#         print(f"Features: {features.shape}")

#     return features

def lbp_pipeline(
    images, 
    block_size,
    P,
    R,
    verbose=False
):

    print("lbp_pipeline")

    features = []

    for image in tqdm(images):

        image_features = extract_lbp_features(
            image,
            block_size=block_size,
            P=P,
            R=R
        )
        features.append(image_features)
    
    features = np.array(features)
    
    if verbose:
        print(f"P: {P}")
        print(f"R: {R}")
        print(f"Features: {features.shape}")

    return features


class LBPTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, block_size, P, R):
        self.block_size = block_size
        self.P = P
        self.R = R

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = lbp_pipeline(X, block_size=self.block_size, P=self.P, R=self.R)
        return np.array(features)


# lbp_sk_pipeline = Pipeline([
#     ('lbp_features', LBPTransformer()),
#     ('svc', SVC(kernel='rbf', gamma='scale', C= 10))
# ])


def full_lbp_pipeline(
    data_path, dataset,
    block_size,
    P,
    R,
    test_size = 0.3, random_state = 42, shuffle = True,
    verbose=False
):
    
    print('--- PREPROCESSING ---')
    images, labels = preprocessing_pipeline(
        data_path=data_path,
        dataset=dataset, 
        verbose=verbose
    ) 

    print('--- LBP ---')
    features = lbp_pipeline(
        images=images, 
        block_size=block_size,
        P=P, 
        R=R, 
        verbose=verbose
    )
    
    print('--- CLASSIF ---')
    train_results, test_results = classifier_pipeline(
        features=features,
        labels=labels,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
        verbose=verbose
    )

    return train_results, test_results



################## HOG ##################


def hog_pipeline(
    images, 
    bins,
    cell_size,
    block_size,
    resize,
    verbose=False
):

    hog_features_images = []

    for image in tqdm(images):

        if resize is not None:
            image = (transform.resize(image, resize)*255).astype(np.uint8)

        magnitude, direction = compute_gradients(image)
        histograms, cells_x, cells_y = compute_histograms(magnitude, direction, cell_size, bins)
        normalized_histograms = block_normalization(histograms, cells_x, cells_y, block_size)
        hog_features_image = compute_hog(normalized_histograms)
        
        # image_features = feature.hog(image, orientations=self.orientations,
        #                             pixels_per_cell=self.pixels_per_cell,
        #                             cells_per_block=self.cells_per_block

        hog_features_images.append(hog_features_image)

    hog_features = np.array(hog_features_images)
    
    if verbose:
        if resize is not None:
            print(f"Resize: {resize}")
        print(f"Bins: {bins}")
        print(f"Cell size: {cell_size}")
        print(f"Block size: {block_size}")
        print(f"Histograms: {histograms.shape[0]}")
        print(f"Features: {hog_features.shape[1]}")


    return hog_features


class HOGTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, orientations, pixels_per_cell, cells_per_block, resize):
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.resize = resize

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # features = [feature.hog(image, orientations=self.orientations,
        #                             pixels_per_cell=self.pixels_per_cell,
        #                             cells_per_block=self.cells_per_block,
        #                             )
        #                 for image in X]
        features = hog_pipeline(
            X, 
            bins=self.orientations,
            cell_size=self.pixels_per_cell,
            block_size=self.cells_per_block,
            resize=self.resize
        )
        return np.array(features)


# hog_sk_pipeline = Pipeline([
#     ('hog_features', HOGTransformer()),
#     ('svc', SVC(kernel='rbf', gamma='scale', C= 10))
# ])


def full_hog_pipeline(
    data_path, dataset,
    bins,
    cell_size,
    block_size,
    resize,
    test_size = 0.3, random_state = 42, shuffle = True,
    verbose=False
):
    
    print('--- PREPROCESSING ---')
    images, labels = preprocessing_pipeline(
        data_path=data_path,
        dataset=dataset, 
        verbose=verbose
    ) 

    print('--- HOG ---')
    features = hog_pipeline(
        images=images, 
        bins=bins, 
        cell_size=cell_size, 
        block_size=block_size,
        resize=resize,
        verbose=verbose
    )
    
    print('--- CLASSIF ---')
    train_results, test_results = classifier_pipeline(
        features=features,
        labels=labels,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
        verbose=verbose
    )

    return train_results, test_results


################## GABOR ##################

def gabor_pipeline(
    images, 
    kernel_size,
    sigma,
    gamma, 
    lmbda,
    psi,
    angles,
    resize,
    flatten=True,
    verbose=False
):

    features = []

    for image in tqdm(images):

        if resize is not None:
            image = (transform.resize(image, resize)*255).astype(np.uint8)

        gabor_image = gabor_process(
            image,
            kernel_size=kernel_size, 
            sigma=sigma,
            gamma=gamma, 
            lmbda=lmbda,
            psi=psi,
            angles=angles
        )
        if flatten:
            image_features = gabor_image.flatten()
        else:
            image_features = gabor_image
        features.append(image_features)

    features = np.array(features)
    
    if verbose:
        if resize is not None:
            print(f"resize: {resize}")
        print(f"kernel size: {kernel_size}")
        print(f"sigma: {sigma}")
        print(f"gamma: {gamma}")
        print(f"lambda: {lmbda}")
        print(f"angles: {angles}")
        print(f"Features: {features.shape}")

    return features


class GaborTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, 
        kernel_size,
        sigma,
        gamma, 
        lmbda,
        psi,
        angles,
        resize
    ):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.gamma = gamma
        self.lmbda = lmbda
        self.psi = psi
        self.angles = angles
        self.resize = resize

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = gabor_pipeline(
            X, 
            kernel_size=self.kernel_size,
            sigma=self.sigma,
            gamma=self.gamma, 
            lmbda=self.lmbda,
            psi=self.psi,
            angles = self.angles,  
            resize = self.resize,
        )
        return np.array(features)


# gabor_sk_pipeline = Pipeline([
#     ('gabor_features', GaborTransformer()),
#     ('svc', SVC(kernel='rbf', gamma='scale', C= 10))
# ])


def full_gabor_pipeline(
    data_path, dataset,
    kernel_size,
    sigma,
    gamma, 
    lmbda,
    psi,
    angles,   
    resize,
    test_size = 0.3, random_state = 42, shuffle=True,
    verbose=False
):
    
    print('--- PREPROCESSING ---')
    images, labels = preprocessing_pipeline(
        data_path=data_path,
        dataset=dataset, 
        verbose=verbose
    ) 

    print('--- GABOR ---')
    features = gabor_pipeline(
        images=images, 
        kernel_size=kernel_size,
        sigma=sigma,
        gamma=gamma, 
        lmbda=lmbda,
        psi=psi,
        angles=angles,  
        resize=resize,
        verbose=verbose
    )
    
    print('--- CLASSIF ---')
    train_results, test_results = classifier_pipeline(
        features=features,
        labels=labels,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
        verbose=verbose
    )

    return train_results, test_results