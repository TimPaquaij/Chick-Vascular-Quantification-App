import cv2
import numpy as np





def process(img, filters):
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8U, kern, borderType=cv2.BORDER_REPLICATE)
        np.maximum(accum, fimg, accum)
    return accum

def adjust_gamma(imgs, gamma=1.0):
    # assert (len(imgs.shape)==4)  #4D arrays
    # assert (imgs.shape[1]==1)  #check the channel is 1
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    new_imgs = np.zeros_like(imgs)
    for i in range(imgs.shape[0]):
        for j in range(imgs.shape[1]):
            new_imgs[i, j] = cv2.LUT(np.array(imgs[i, j], dtype=np.uint8), table)
    return new_imgs

def build_filters2(sigma=1, YLength=10):
    filters = []
    widthOfTheKernel = np.ceil(np.sqrt((6 * np.ceil(sigma) + 1) ** 2 + YLength ** 2))
    if np.mod(widthOfTheKernel, 2) == 0:
        widthOfTheKernel = widthOfTheKernel + 1
    widthOfTheKernel = int(widthOfTheKernel)
    # print(widthOfTheKernel)
    for theta in np.arange(0, np.pi, np.pi / 16):
        # theta = np.pi/4
        matchFilterKernel = np.zeros((widthOfTheKernel, widthOfTheKernel), dtype=np.float)
        for x in range(widthOfTheKernel):
            for y in range(widthOfTheKernel):
                halfLength = (widthOfTheKernel - 1) / 2
                x_ = (x - halfLength) * np.cos(theta) + (y - halfLength) * np.sin(theta)
                y_ = -(x - halfLength) * np.sin(theta) + (y - halfLength) * np.cos(theta)
                if abs(x_) > 3 * np.ceil(sigma):
                    matchFilterKernel[x][y] = 0
                elif abs(y_) > (YLength - 1) / 2:
                    matchFilterKernel[x][y] = 0
                else:
                    matchFilterKernel[x][y] = -np.exp(-.5 * (x_ / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma)
        m = 0.0
        for i in range(matchFilterKernel.shape[0]):
            for j in range(matchFilterKernel.shape[1]):
                if matchFilterKernel[i][j] < 0:
                    m = m + 1
        mean = np.sum(matchFilterKernel) / m
        for i in range(matchFilterKernel.shape[0]):
            for j in range(matchFilterKernel.shape[1]):
                if matchFilterKernel[i][j] < 0:
                    matchFilterKernel[i][j] = matchFilterKernel[i][j] - mean
        filters.append(matchFilterKernel)

    return filters

def grayStretch(img, m=60.0/255, e=8.0):
    k = np.zeros(img.shape, np.float)
    ans = np.zeros(img.shape, np.float)
    mx = np.max(img)
    mn = np.min(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            k[i][j] = (float(img[i][j]) - mn) / (mx - mn)
    eps = 0.01
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            ans[i][j] = 1 / (1 + (m / (k[i][j] + eps)) ** e) * 255
    ans = np.array(ans, np.uint8)
    return ans
