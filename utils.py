import numpy as np
import cv2
import SimpleITK as sitk
import matplotlib.pyplot as plt
from  pathlib import Path

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


def getFileNames(index, config, sp):
    dataRoot = Path(config['data']['dataRoot'])
    fileNameTemplate = config['data']['fileNameTemplate']
    fileTSeedTemplate = config['data']['fileTSeedTemplate']
    fileCrackTemplate = config['data']['fileCrackTemplate']
    fileSeedTemplate = config['data']['fileSeedTemplate']
    templates = [fileNameTemplate, fileTSeedTemplate, fileCrackTemplate]

    fileName, fileTSeed, fileCrack = list(map(lambda x: dataRoot / x.format(index), templates))
    if sp == 'file':
        return fileName
    elif sp == 'tseed':
        return  fileTSeed
    elif sp == 'tcrack':
        return fileCrack
    elif sp == 'seed':
        return fileSeedTemplate
    else:
        raise Exception('bad sp param')

