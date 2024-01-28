# import sys
# sys.path.append('../src')

from data_manager import get_image_paths, get_images, get_labels

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


def get_score(clf, X, y, train=True):

    pred = clf.predict(X)
    
    metrics = {
        'accuracy': accuracy_score(y, pred),
        'clf_report': classification_report(y, pred),
        'cm': confusion_matrix(y, pred)
    }

    return pred, metrics


def print_score(y_true, y_pred, train=True, add_cr=False, add_cm=False):

    if train:
        print('================================================')
        print("Train Result:")
    else:
        print('================================================')
        print("Test Result:")
    
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy Score: {accuracy * 100:.2f}%")

    if add_cr:
        clf_report = classification_report(y_true, y_pred)
        print("_______________________________________________")
        print(f"Classification report:\n{clf_report}")

    if add_cm:
        cm = confusion_matrix(y_true, y_pred)
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {cm}\n")
    

def classifier_pipeline(features, labels, verbose=False):

    X_train, X_test, y_train, y_test = train_test_split(
        images, labels,
        test_size=0.3, random_state=42, shuffle=True
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


def lbp_pipeline(
    images, 
    P = 8,
    R = 1,
    resize = None,
    verbose=False
):

    features = []

    for image in tqdm(images):

        if resize is not None:
            image = transform.resize(image, resize)

        lbp_image = feature.local_binary_pattern(
            image,
            P=P,
            R=R
        )
        image_features = lbp_image.flatten()
        features.append(image_features)

    features = np.array(features)
    
    if verbose:
        print(f"P: {P}")
        print(f"R: {R}")
        print(f"Features: {features.shape}")

    return features


class LBPTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, P=3, R=1, resize=None):
        self.P = P
        self.R = R
        self.resize = resize

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = lbp_pipeline(X, P=self.P, R=self.R, resize=self.resize)
        return np.array(features)


# lbp_sk_pipeline = Pipeline([
#     ('lbp_features', LBPTransformer(P=3, R=1, resize=(32,32))),
#     ('svc', SVC(kernel='rbf', gamma='scale', C= 10))
# ])


def full_lbp_pipeline(
    data_path, 
    dataset,
    P = 8,
    R = 3,
    resize = (32,32),
    test_size = 0.3,
    random_state = 42,
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
        P=P, 
        R=R, 
        resize=resize,
        verbose=verbose
    )
    
    print('--- CLASSIF ---')
    train_results, test_results = classifier_pipeline(
        features=features,
        labels=labels,
        verbose=verbose
    )

    return train_results, test_results



### HOG ###


def hog_pipeline(
    images, 
    bins = 9,
    cell_size = (8, 8),
    block_size = (2, 2),
    verbose=False
):

    hog_features_images = []

    for image in tqdm(images):

        magnitude, direction = compute_gradients(image)
        histograms, cells_x, cells_y = compute_histograms(magnitude, direction, cell_size, bins)
        normalized_histograms = block_normalization(histograms, cells_x, cells_y, block_size)
        hog_features_image = compute_hog(normalized_histograms)

        hog_features_images.append(hog_features_image)

    hog_features = np.array(hog_features_images)
    
    if verbose:
        print(f"Bins: {bins}")
        print(f"Cell size: {cell_size}")
        print(f"Block size: {block_size}")
        print(f"Histograms: {histograms.shape[0]}")
        print(f"Features: {hog_features.shape[1]}")

    return hog_features


class HOGTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        hog_features = [feature.hog(image, orientations=self.orientations,
                                    pixels_per_cell=self.pixels_per_cell,
                                    cells_per_block=self.cells_per_block)
                        for image in X]
        return np.array(hog_features)


# hog_sk_pipeline = Pipeline([
#     ('hog_features', HOGTransformer()),
#     ('svc', SVC(kernel='rbf', gamma='scale', C= 10))
# ])


def full_hog_pipeline(
    data_path, 
    dataset,
    bins = 9,
    cell_size = (8, 8),
    block_size = (2, 2),
    test_size = 0.3,
    random_state = 42,
    verbose=False
):
    
    print('--- PREPROCESSING ---')
    images, labels = preprocessing_pipeline(
        data_path=data_path,
        dataset=dataset, 
        verbose=verbose
    ) 

    print('--- HOG ---')
    hog_features = hog_pipeline(
        images=images, 
        bins=bins, 
        cell_size=cell_size, 
        block_size=block_size,
        verbose=verbose
    )
    
    print('--- CLASSIF ---')
    train_results, test_results = classifier_pipeline(
        features=hog_features,
        labels=labels,
        verbose=verbose
    )

    return train_results, test_results