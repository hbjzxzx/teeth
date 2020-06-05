import numpy as np
import cv2
from utils import max_min_normal, getFileNames, readFromNii
import lv_set.drlse_algo as drlse
from tqdm import tqdm
import SimpleITK as sitk

def main():
    from config import  config
    cfg = config('configs/config.yaml')
    doLevelSet(cfg)

def doLevelSet(config):
    picRange = config['data']['levelset']['range']
    pbTotal = tqdm(picRange)
    for i in pbTotal:
        tseedFileName = getFileNames(i, config, 'tseed')
        filename = getFileNames(i, config, 'file')
        pbTotal.update()
        pbTotal.set_description(f'handle {filename}')
        images = readFromNii(str(filename))
        inits = readFromNii(str(tseedFileName))
        result = []
        pbSingle = tqdm(zip(images, inits))
        for id, (img, init) in enumerate(pbSingle):
            pbSingle.update()
            pbSingle.set_description(f'handle slices {id}')
            if (np.max(init) == 0):
                result.append(init)
                continue
            postphi, phi = levelSet(img, init)
            result.append(postphi)
        result = np.array(result)
        fileSeedName = getFileNames(i, config, 'seed')
        sitk.WriteImage(result, str(fileSeedName))

def postProcess(flevel):
    l = np.zeros_like(flevel)
    l[flevel < 0] = 1
    l = l.astype(np.uint8)
    re, _ = cv2.findContours(l, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(l)
    for i in range(len(re)):
        cv2.drawContours(mask, re, i, 1, cv2.FILLED)

    return mask

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
    lmda = 6  # coefficient of the weighted length term L(phi)
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
    pb = tqdm(range(iter_outer))
    for iteid, n in enumerate(pb):
        pb.update()
        pb.set_description(f'iter: {iteid+1}/{iter_outer}')
        phi = drlse.drlse_edge(phi, g, lmda, mu, alfa, epsilon, timestep, iter_inner, potentialFunction)

    # refine the zero level contour by further level set evolution with alfa=0
    alfa = 0
    iter_refine = 10
    phi = drlse.drlse_edge(phi, g, lmda, mu, alfa, epsilon, timestep, iter_refine, potentialFunction)
    postPhi = postProcess(phi)
    return postPhi, phi

if __name__ == '__main__':
    main()
