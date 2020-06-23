import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from models import regNets
from config import config
from data import class2set, imageSet
from utils import *
from AtNets import *
import time
import sys

classes = ['healthy', 'crack']
globalStep = 0

class Collectors(object):
    def clear(self):
        raise NotImplemented()

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
    def __init__(self,name, topK=1):
        self.AT = []
        self.name = name
        self.topK = topK
    
    def clear(self):
        self.AT.clear()

    def add_pred(self, ats:torch.Tensor):
        at = ats[:self.topK].detach().cpu()
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

def main():
    #net = 'resnet'
    #net = 'inceptionv3'
    torch.backends.cudnn.benchmark=True
    netConfig = sys.argv[1] + '.yaml'
    configRoot = Path('configs')
    dataConfigFileName = Path('dataconfig.yaml')
    #netConfigFileName = Path(f'{net}.yaml')
    netConfigFileName = configRoot/Path(netConfig)
    assert netConfigFileName.exists(), f'file {str(netConfigFileName)} not exist'
    cfg = config(str(configRoot/dataConfigFileName))
    cfg.mergeWith(str(netConfigFileName))
    net = cfg['train']['netname']

    # data
    batchSize = cfg['train']['batch_size']
    shuffle = cfg['train']['shuffle']
    numWorkers = cfg['train']['num_worker']
    balance = cfg['train']['balance']
    splitOnImage = cfg['data']['splitedOnEntity']    

    
    # train
    netname = cfg['train']['netname']
    num_epochs = cfg['train']['epoch']
    device = cfg['train']['device']
    device = torch.device(device)
    outputDir = Path(cfg['train']['output_dir'])
    netoutdir = Path(netname)

    thisRunName = cfg['train']['session_dir']
    if thisRunName == 'auto':
        thisRunName = getSessionName(cfg)
        print(f'auto generate Session name: {thisRunName}')
    thisRunName = Path(thisRunName)

    saveRoot = outputDir/netoutdir/thisRunName
    if saveRoot.exists():
        val = input(f'remove {str(saveRoot)} and continue ? y or n')
        if val == 'y':
            shutil.rmtree(str(saveRoot))
        else:
            raise Exception("Stop for protect the existing data")
    tfLog = Path('tensorboard')

    writer = SummaryWriter(str(saveRoot/tfLog))
    save_step = cfg['train']['save_step']
    info_step = cfg['train']['info_step']
    test_step = cfg['train']['test_step']


    # loss
    lossType  = cfg['loss']['type']
    posWeight = cfg['loss']['posWeight']
    if lossType == 'CE':
        Clscriterion = ClsWeightLoss(posWeight) 
        print(f'using loss: CE ',end=' ')
    elif lossType == 'FOCAL':
        gamma = cfg['loss']['gamma']
        Clscriterion = FocalLoss(posWeight) 
        print(f'using loss: FOCALLoss gamma:{gamma}',end=' ')
    print(f'powWeight:{posWeight}')

    
    if cfg['data']['splitedOnEntity']:
        TrainDataSet = class2set(cfg, isTrain=True)
        TestDataSet = class2set(cfg, isTrain=False)
    else:
        dset = imageSet(cfg) 
        TrainDataSet, TestDataSet = dset.genTrainTest()

    if balance:
        prate = TrainDataSet.prate
        weights = []
        for _, l in TrainDataSet:
            weights.append(1-prate if l==1 else prate)
        trainSampler = WeightedRandomSampler(weights, len(TrainDataSet), replacement=True)

        TrainDataloader = DataLoader(TrainDataSet,
                                    batch_size=batchSize,
                                    num_workers=numWorkers,
                                        sampler=trainSampler)
        print(f"using balance data")
    else:
        TrainDataloader  = DataLoader(TrainDataSet,
                                    batch_size=batchSize,
                                    shuffle=shuffle,
                                    num_workers=numWorkers)

    TestDataloader = DataLoader(TestDataSet,
                                    batch_size=batchSize,
                                    shuffle=shuffle,
                                    num_workers=numWorkers)

    print(f'loading mode:{net}')
    model = regNets[net](config=cfg)
    if cfg['optim']['type'] == 'SGD':
        lr =cfg['optim']['lr']
        m = cfg['optim']['momentum']
        wd = cfg['optim']['weight_decay']
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=m, weight_decay=wd)
        print(f'using optim SGD lr:{lr} momentum:{m} weight_decay:{wd}')

    dataiter = iter(TrainDataloader)
    images, labels = dataiter.next()
    imgGrid = torchvision.utils.make_grid(images)
    matplotlib_imshow(imgGrid, one_channel=True)
    writer.add_image('sample images', imgGrid)
    writer.add_graph(model, images)
    model.to(device)
    global globalStep
    globalStep = 0

    try:
        for e in range(num_epochs):
            s = time.time()
            trainCls(model, optimizer, Clscriterion, TrainDataloader, TestDataloader, 
                                device, writer, e+1, 
                                save_step, info_step, test_step, saveRoot, batchSize)
            total = time.time() - s
            e1time = total/60.0
            left = (num_epochs - e -1)*e1time/60.0
            print(f'epoch time ：{total/60.0:6.3f}min, left:{left:6.3f}hours')
        torch.save(model.state_dict(), str(saveRoot/Path('model_final.pth')))
    except KeyboardInterrupt:
        torch.save(model.state_dict(), str(saveRoot/Path('model_final_keyStop.pth')))

def trainCls(model, opt, lossFunc, trainDataLoader, testDataLoader, device, writer, epoch, save_step, info_step, test_step, saveRoot, batch):
    global globalStep 
   
    clsRec = ClsReCollector()
    clsLossReC = LossReCollector('cls Loss')
    pnCe = PNCollector()

    collectors = [clsRec, clsLossReC, pnCe]
    
    model.train()
    info_timeS = time.time()
    for i, data in enumerate(trainDataLoader):
        globalStep += 1
        inputs, labels = data[0].to(device), data[1].to(device)
        opt.zero_grad()
        out = model(inputs)
        loss = lossFunc(out, labels)
        loss.backward()
        opt.step()

        clsRec.add_preds_labels(out, labels)
        clsLossReC.add(loss)         
        pnCe.add(target=labels)

        if globalStep % save_step == save_step-1:
            ckp = Path(f'checkpoint_{epoch}_{globalStep+1}.pth')
            torch.save(model.state_dict(), str(saveRoot/ckp))

        if globalStep % info_step  == info_step-1:
            useTime = time.time() - info_timeS
            print("epoch:{:2d}, step:{:4d} totalStep:{:6d}".format(epoch, i+1, globalStep), end='  ')
            print("{}".format(clsLossReC), end=' ')
            print("{}".format(pnCe), end=' ')
            print(f"{useTime:5.4f}s for {info_step} step, {useTime/(info_step*batch):5.4f}s per image")

            test_probs, gt = clsRec.get_result()
            writer.add_pr_curve('healthy train', 1-gt , 1 - test_probs, global_step=globalStep)
            writer.add_pr_curve('crack train', gt , test_probs, global_step=globalStep)

            writer.add_scalar('training cls loss', clsLossReC.get(), global_step=globalStep)

            for c in collectors:
                c.clear()
            torch.cuda.empty_cache()
            info_timeS = time.time()

        if globalStep % test_step == test_step -1:
            stime = time.time()
            print('start tesing:  ',end='')
            testCls(model, lossFunc, testDataLoader, device, writer, globalStep)
            etime = time.time() - stime
            print(f'testing done time:{etime:5.2f}s')

def testCls(model:torch.nn.Module, lossFunc, testDataLoader, device, writer, step):
    model.eval()
    clsRec = ClsReCollector()
    lossRec = LossReCollector('Testing Loss') 
    with torch.no_grad():
        for data in testDataLoader:
            inputs, labels = data[0].to(device), data[1].to(device)
            output = model(inputs)
            ls = lossFunc(output, labels)

            clsRec.add_preds_labels(output, labels)
            lossRec.add(ls)

    test_probs, gt = clsRec.get_result()
    tloss = lossRec.get()
    writer.add_scalar('Test cls loss', tloss, global_step=step)
    writer.add_pr_curve('Test Healthy PR', 1-gt , 1 - test_probs, global_step=step)
    writer.add_pr_curve('Test Crack PR', gt , test_probs, global_step=step)
    model.train()


if __name__=='__main__':
    main()
