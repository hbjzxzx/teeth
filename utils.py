import numpy as np
import cv2
import SimpleITK as sitk
import matplotlib.pyplot as plt
from  pathlib import Path
import lv_set.drlse_algo as drlse
from tqdm import tqdm

def normalImage(image, norm):
    if norm == 8:
        image = np.uint8(max_min_normal(image))
    elif norm == 16:
        image = np.uint16(max_min_normal(image, 16))
    else:
        raise Exception('error input normal, must be 8 or 16')
    return image

def resizeImage(image, h, w):
    image = cv2.resize(image, (h, w))
    return image

def readFromNii(path):
    image = sitk.ReadImage(path)
    image = sitk.GetArrayFromImage(image)
    return image

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

def max_min_normal(im, maxd=8):
    maxd = 2 ** maxd -1
    minI = np.min(im)
    im -= minI
    maxI = np.max(im)
    im = im/maxI * maxd
    return im

def doLevelSet(config):
    picRange = config['data']['levelset']['range']
    for i in picRange:
        tseedFileName = getFileNames(i, config, 'tseed')
        filename = getFileNames(i, config, 'file')
        images = readFromNii(str(filename))
        inits = readFromNii(str(tseedFileName))
        result = []
        for img, init in zip(images, inits):
            if (np.max(init) == 0):
                result.append(init)
                continue
            phi = levelSet(img, init)




def getFileNames(index, config, sp):
    dataRoot = Path(config['data']['dataRoot'])
    fileNameTemplate = config['data']['fileNameTemplate']
    fileTSeedTemplate = config['data']['fileTSeedTemplate']
    fileCrackTemplate = config['data']['fileCrackTemplate']
    templates = [fileNameTemplate, fileTSeedTemplate, fileCrackTemplate]

    fileName, fileTSeed, fileCrack = list(map(lambda x: dataRoot / x.format(index), templates))
    if sp == 'file':
        return fileName
    elif sp == 'tseed':
        return  fileTSeed
    elif sp == 'tcrack':
        return fileCrack
    else:
        raise Exception('bad sp param')


def levelSet(image, seed):
    im = np.copy(image)
    im = max_min_normal(im).astype('uint8')
    im = cv2.GaussianBlur(im, (5, 5), 0)
    im = cv2.medianBlur(im, 5)

    init = np.copy(seed)
    init[init == 1] = -2
    init[init == 0] = 2

    # parameters
    timestep = 1  # time step
    mu = 0.2 / timestep  # coefficient of the distance regularization term R(phi)
    iter_inner = 10
    iter_outer = 100
    lmda = 4  # coefficient of the weighted length term L(phi)
    alfa = -12  # coefficient of the weighted area term A(phi)
    epsilon = 0.1  # parameter that specifies the width of the DiracDelta function

    # sigma = 0.0         # scale parameter in Gaussian kernel
    # img_smooth = filters.gaussian_filter(img, sigma)    # smooth image by Gaussian convolution
    [Iy, Ix] = np.gradient(im)
    f = np.square(Ix) + np.square(Iy)
    # gray_lap = cv2.Laplacian(img,cv2.CV_16S,ksize = 3)
    g = 1 / (1 + np.power(f, 0.8))  # edge indicator function.

    # initialize LSF as binary step function
    # generate the initial region R0 as two rectangles
    phi = init

    potential = 2
    if potential == 1:
        potentialFunction = 'single-well'  # use single well potential p1(s)=0.5*(s-1)^2, which is good for region-based model
    elif potential == 2:
        potentialFunction = 'double-well'  # use double-well potential in Eq. (16), which is good for both edge and region based models
    else:
        potentialFunction = 'double-well'  # default choice of potential function

    # start level set evolution
    for n in tqdm(range(iter_outer)):
        phi = drlse.drlse_edge(phi, g, lmda, mu, alfa, epsilon, timestep, iter_inner, potentialFunction)

    # refine the zero level contour by further level set evolution with alfa=0
    alfa = 0
    iter_refine = 10
    phi = drlse.drlse_edge(phi, g, lmda, mu, alfa, epsilon, timestep, iter_refine, potentialFunction)

    return phi
