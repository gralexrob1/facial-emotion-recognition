import numpy as np
from skimage import feature

def extract_lbp_features(image, block_size, P, R):

    # Divide the image into small regions (blocks)
    regions = [
        image[y:y+block_size[1], x:x+block_size[0]] 
        for y in range(0, image.shape[0], block_size[1]) 
        for x in range(0, image.shape[1], block_size[0])
    ]

    #  Calculate LBP histogram for each region
    lbp_histograms = []
    for region in regions:
        lbp = feature.local_binary_pattern(region, P=P, R=R, method="uniform")
        hist, _ = np.histogram(lbp.ravel(), bins=50)
        lbp_histograms.extend(hist)

    # Concatenate histograms into a single feature vector
    feature_vector = np.array(lbp_histograms)

    return feature_vector
