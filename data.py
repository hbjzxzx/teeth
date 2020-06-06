import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import SimpleITK as sitk
import cv2
from pathlib import Path
from utils import *


class baseset(Dataset):
    def __init__(self, config, isTrain=True):
        self.h = config['image']['height']
        self.w = config['image']['width']
        self.norm = config['image']['normal']
        gk = config['image']['gKernelSize']
        self.gkSize = (gk, gk)
        self.medSize = config['image']['medKernelSize']
        self.picRange = range(*config['data']['trainRange']) if isTrain else range(*config['data']['testRange'])
        self.istrain = isTrain
        self.config = config
        # map from index to data
        self.rawdataDic = {}
        for i in self.picRange:
            fileName = getFileNames(i, self.config, 'file')
            self.rawdataDic[i] = readFromNii(str(fileName))

        self.gindex2lindex = {}
        cnt = 0
        for p in self.picRange:
            for lindex in range(len(self.rawdataDic[p])):
                self.gindex2lindex[cnt] = (p, lindex)
                cnt += 1
        self.len = cnt

    def __len__(self):
        return  self.len

    def __getitem__(self, item):
        raise Exception('not implement')

    def __loadLabel(self):
        raise Exception('not implemnt')


class class2set(baseset):
    def __init__(self, config, isTrain=True):
        super().__init__(config, isTrain)
        self.__loadLabel()
    def __loadLabel(self):
        self.rawCmaskDic = {}
        for i in self.picRange:
            fileCrackName = getFileNames(i, self.config, 'tcrack' )
            crackMask = readFromNii(str(fileCrackName))
            crackMask[crackMask>0] = 1
            self.rawCmaskDic[i] = crackMask

        self.gindexCrackLabel = []
        for gindex in range(self.len):
            p, lindex = self.gindex2lindex[gindex]
            cmask = self.rawCmaskDic[p][lindex]
            iscrack = 1 if np.max(cmask) != 0 else 0
            self.gindexCrackLabel.append(iscrack)

        p = sum(self.gindexCrackLabel)
        n = self.len - p
        self.prate = p/self.len
        print("pLabel:{} nLabel:{}".format(p, n))

    def getCmask(self, index):
        p, lindex = self.gindex2lindex[index]
        cmask = self.rawCmaskDic[p][lindex]
        cmask = resizeImage(cmask, self.h, self.w)
        return cmask

    def getNormData(self, index):
        p, lindex = self.gindex2lindex[index]
        image = self.rawdataDic[p][lindex]
        image = normalImage(image, self.norm)
        if self.config['image']['gaussFilter']:
            image = cv2.GaussianBlur(image, self.gkSize, 0)
        if self.config['image']['medianFilter']:
            image = cv2.medianBlur(image, self.medSize)
        image = resizeImage(image, self.h, self.w)
        return image

    def __getitem__(self, item):
        if item >= self.len:
            raise StopIteration
        cmask = self.getCmask(item)
        # 0 for none Crack 1 for exist Crack
        isCrack = self.gindexCrackLabel[item]
        label = [0, 1] if isCrack else [1, 0]
        image = self.getNormData(item)
        image = np.float32(image)
        image = np.stack([image]*3, axis=0)
        return image, label
