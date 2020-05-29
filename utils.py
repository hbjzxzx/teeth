import numpy as np
import cv2
import SimpleITK as sitk
import matplotlib.pyplot as plt
from  pathlib import Path
from levelset import levelSet
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
            phi, _ = levelSet(img, init)
            result.append(phi)





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

