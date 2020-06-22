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
        self.picRangeOfLevelSet = range(*config['data']['levelset']['range'])
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
        print("pLabel:{} nLabel:{} prate:{:.3f}".format(p, n, self.prate))

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
        label = 1 if isCrack else 0 
        image = self.getNormData(item)
        image = np.float32(image)
        image = np.stack([image]*3, axis=0)
        return torch.from_numpy(image), torch.from_numpy(np.array(label))


class class2setWithATMask(class2set):

    def __init__(self, config, isTrain=True):
        super().__init__(config, isTrain)
        self.__loadATMask()


    def __loadATMask(self):
        self.rawATMask = {}
        cnt = 0
        for i in self.picRange:
            fileNameSeed = getFileNames(i, self.config, 'seed' )
            try:
                crackMask = readFromNii(str(fileNameSeed))
            except Exception:
                continue
            self.rawATMask[i] = crackMask
            cnt += 1
        print(f'load AT instance {cnt}')


    def getMachATMask(self, index):
        p, lindex = self.gindex2lindex[index]
        if p in self.rawATMask.keys():
            atmask = self.rawATMask[p][lindex]
            atmask = resizeImage(atmask, self.h, self.w)
            atmask = resizeImage(atmask, 16, 16)
            atmask = atmask.astype(np.int32)
            #atmask = atmask - np.min(atmask) 
            #atmask = atmask / np.max(atmask)
            assert np.max(atmask) == 1 or np.max(atmask) == 0
        else:
            atmask = np.ones((16,16), dtype=np.int32) * -1
        return atmask

    def __getitem__(self, item):
        image, label = super(class2setWithATMask, self).__getitem__(item)
        Atmask = self.getMachATMask(item)
        return image, label, torch.from_numpy(Atmask)
