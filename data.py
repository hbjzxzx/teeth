import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import SimpleITK as sitk
import cv2
from pathlib import Path
from utils import *
import os
from sklearn.model_selection import train_test_split

class imageSpDataSet(Dataset):
    def __init__(self, pics, labels, config):
        self.pics = pics
        self.labels = labels
    
        self.config = config
        self.h = config['image']['height']
        self.w = config['image']['width']
        self.norm = config['image']['normal']
        gk = config['image']['gKernelSize']
        self.gkSize = (gk, gk)
        self.medSize = config['image']['medKernelSize']

        self.len = len(self.pics)
        self.p = sum(labels)
        n = self.len - self.p
        self.prate = self.p/self.len 
        
    def __len__(self):
        return self.len 
    
    def __getitem__(self, item):
        im = self.getNormData(item) 
        crack = self.labels[item]
        im = torch.from_numpy(im)
        crack = torch.from_numpy(np.array(crack))
        im = np.float32(im)
        im = np.stack([im]*3, axis=0)
        return im, crack
        
    def getNormData(self, index):
        image = self.pics[index]
        image = normalImage(image, self.norm)
        if self.config['image']['gaussFilter']:
            image = cv2.GaussianBlur(image, self.gkSize, 0)
        if self.config['image']['medianFilter']:
            image = cv2.medianBlur(image, self.medSize)
        image = resizeImage(image, self.h, self.w)
        return image

class imageSet(object):
    def __init__(self, config):
        self.config = config
        self.test_size = config['data']['splitedImagesRate']
        self.randomSeed = config['data']['splitedRandomSeed']
        self._loadData(self.config)
    def _loadData(self, config):
        self.picRange = config['data']['splitedImagesRange']

        self.config = config
        # map from index to data
        self.rawPic = [] 
        self.labels = []

        for i in self.picRange:
            fileName = getFileNames(i, self.config, 'file')
            images = readFromNii(str(fileName))
            for image in images:
                self.rawPic.append(image)

            fileCrackName = getFileNames(i, self.config, 'tcrack' )
            crackMasks = readFromNii(str(fileCrackName))
            for cmask in crackMasks:
                iscrack = 1 if np.max(cmask) != 0 else 0
                self.labels.append(iscrack)
        p = sum(self.labels)
        n = len(self.labels) - p
        assert len(self.labels) == len(self.rawPic), 'Error, labels number can not match pictures'
        self.len = len(self.labels)
        self.prate = p/self.len
        print("pLabel:{} nLabel:{} prate:{:.3f}".format(p, n, self.prate))

    def getPNCnt(self, label):
        p = sum(label)
        n = len(label) - p
        prate = p/(p+n)
        return p, n, prate

    def genTrainTest(self):
        pic_train, pic_test, label_train, label_test = train_test_split(self.rawPic, self.labels, 
                                                                        test_size=self.test_size, random_state=self.randomSeed)
        trainDataSet = imageSpDataSet(pic_train, label_train, self.config)
        print('train Data p:{} n:{}  prate{:.2f}'.format(*self.getPNCnt(label_train)))
        testDataSet = imageSpDataSet(pic_test, label_test, self.config)
        print('test Data p:{} n:{}  prate{:.2f}'.format(*self.getPNCnt(label_test)))

        return trainDataSet, testDataSet

class baseset(Dataset):
    def __init__(self, config, isTrain=True):
        self.h = config['image']['height']
        self.w = config['image']['width']
        self.norm = config['image']['normal']
        gk = config['image']['gKernelSize']
        self.gkSize = (gk, gk)
        self.medSize = config['image']['medKernelSize']
        self.picRange = config['data']['trainRange'] if isTrain else config['data']['testRange']
        self.picRangeOfLevelSet = config['data']['levelset']['range']
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
