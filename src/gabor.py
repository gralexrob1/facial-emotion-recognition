import numpy as np


def gabor_filter(kernel_size, sigma, gamma, lmbda, psi, angle):
    # get half size
    d = kernel_size // 2

    # prepare kernel
    gabor = np.zeros((kernel_size, kernel_size), dtype=np.float32)

    # each value
    for y in range(kernel_size):
        for x in range(kernel_size):
            # distance from center
            px = x - d
            py = y - d

            # degree -> radian
            theta = angle / 180. * np.pi

            # kernel x
            _x = np.cos(theta) * px + np.sin(theta) * py

            # kernel y
            _y = -np.sin(theta) * px + np.cos(theta) * py

            # fill kernel
            gabor[y, x] = np.exp(-(_x**2 + gamma**2 * _y**2) / (2 * sigma**2)) * np.cos(2*np.pi*_x/lmbda + psi)

    # kernel normalization
    gabor /= np.sum(np.abs(gabor))

    return gabor


def gabor_filtering(image, kernel_size, sigma, gamma, lmbda, psi, angle):
    # get shape
    H, W = image.shape

    # padding
    image = np.pad(image, (kernel_size//2, kernel_size//2), 'edge')

    # prepare out image
    out = np.zeros((H, W), dtype=np.float32)

    # get gabor filter
    gabor = gabor_filter(
        kernel_size=kernel_size, 
        sigma=sigma, 
        gamma=gamma, 
        lmbda=lmbda,
        psi=psi, 
        angle=angle
    )

    # filtering
    for y in range(H):
        for x in range(W):
            out[y, x] = np.sum(image[y : y + kernel_size, x : x + kernel_size] * gabor)

    out = np.clip(out, 0, 255)
    out = out.astype(np.uint8)

    return out


# Use 6 Gabor filters with different angles to perform feature extraction on the image
def gabor_process(
    image, 
    kernel_size=11,
    sigma=1.5,
    gamma=1.2, 
    lmbda=3,
    psi=0,
    angles = [0,30,60,90,120,150]
):
    
    H, W = image.shape
    out = np.zeros([H, W], dtype=np.float32)

    # each angle
    for i, angle in enumerate(angles):
    
        # gabor filtering
        _out = gabor_filtering(
            image, 
            kernel_size=kernel_size, 
            sigma=sigma, 
            gamma=gamma, 
            lmbda=lmbda, 
            psi=psi, 
            angle=angle
        )
         
        # add gabor filtered image
        out += _out
        
    # scale normalization
    out = out /out.max()*255
    out = out.astype(np.uint8)

    return out