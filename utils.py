import numpy as np
import cv2
import SimpleITK as sitk
import matplotlib.pyplot as plt
from  pathlib import Path
import shutil
import torch

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
    templates = [fileNameTemplate, fileTSeedTemplate, fileCrackTemplate, fileSeedTemplate]

    fileName, fileTSeed, fileCrack, fileSeed = list(map(lambda x: dataRoot / x.format(index), templates))
    if sp == 'file':
        return fileName
    elif sp == 'tseed':
        return  fileTSeed
    elif sp == 'tcrack':
        return fileCrack
    elif sp == 'seed':
        return fileSeed
    else:
        raise Exception('bad sp param')


class Collectors(object):
    def clear(self):
        raise NotImplementedError('base class should not be used')

class ClsReCollector(Collectors):
    def __init__(self):
        self.rawPreds = []
        self.rawLabels = []
        self.cnt = 0 
    
    def add_preds_labels(self, preds:torch.Tensor, labels:torch.Tensor):
        predPro = torch.softmax(preds.detach(), dim=1)
        predPro = predPro[:,1]
        self.rawPreds.append(predPro.cpu())
        self.rawLabels.append(labels.cpu())
        self.cnt += 1 
        
    def get_result(self):
        Pred = torch.cat(self.rawPreds)
        Label = torch.cat(self.rawLabels)
        return Pred, Label  
    
    def clear(self):
        self.rawLabels.clear()
        self.rawPreds.clear()
        self.cnt = 0


class LossReCollector(Collectors):
    def __init__(self, name):
        self.lossName = name
        self.lossAcc = 0.0
        self.cnt = 0
    

    def add(self, loss:torch.Tensor):
        self.lossAcc += loss.detach().item()
        self.cnt += 1
    
    def get(self):
        if self.cnt == 0:
            return -1
        else:
            return self.lossAcc / self.cnt
    
    def clear(self):
        self.lossAcc = 0.0
        self.cnt = 0
    
    def __str__(self):
        return f'{self.lossName:5s}:{self.get():.4f}'


class ATReCollector(Collectors):
    def __init__(self, topK=1):
        self.AT = []
        self.topK = topK
    
    def clear(self):
        self.AT.clear()

    def add_pred(self, ats:torch.Tensor):
        at = ats[:self.topK].detach().cpu()
        at = torch.softmax(at, dim=1)
        at = at.rpn.unsqueeze(1).repeat(1,3,1,1)
        self.AT.append(at)

    def get(self):
        return torch.cat(self.AT)


class PNCollector(Collectors):
    def __init__(self):
        self.pos = 0
        self.neg = 0
    
    def clear(self):
        self.pos = 0
        self.neg = 0
    
    def add(self, target:torch.Tensor):
        batch = len(target)
        p = torch.sum(target).detach().cpu().item()
        self.pos += p 
        self.neg += (batch - p)

    def __str__(self):
        return f'posIns:{self.pos:>3} negIns {self.neg:>3}' 

def getSessionName(config):
    balance = 'balance_' + ('On' if config['train']['balance'] else 'Off')
    SplitOn = 'splitedOn_' + ('Entity' if config['data']['splitedOnEntity'] else 'images_{}'.format(config['data']['splitedImagesRate']))
    PreTrainOn = 'PreTrain_' + ('On' if config['train']['pre_train'] else 'Off')
    
    lossInfo = ('CE' if config['loss']['type']=='CE' else 'FocalLoss_gamma{}'.format(config['loss']['gamma'])) + 'posWeight_{}'.format(config['loss']['posWeight'])
    optiInfo = 'learnRate_{}'.format(config['optim']['lr']) + 'weight_decay_{}'.format(config['optim']['weight_decay']) 
    strs = [balance, SplitOn, PreTrainOn, lossInfo, optiInfo]
    
    return '_'.join(strs) 
