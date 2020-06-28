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
from data import class2set, imageSet, class2setWithATMask
from utils import *
from AtNets import *
import time
import sys

classes = ['healthy', 'crack']
globalStep = 0

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
    useATLabels = cfg['train']['useAtLabel']
    
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
    print(f'posWeight:{posWeight}')

    
    if cfg['data']['splitedOnEntity']:
        if useATLabels:
            TrainDataSet = class2setWithATMask(cfg, isTrain=True)
            TestDataSet = class2setWithATMask(cfg, isTrain=False)
        else:
            TrainDataSet = class2set(cfg, isTrain=True)
            TestDataSet = class2set(cfg, isTrain=False)
    else:
        if useATLabels:
            raise NotImplementedError('can use image splited when turn on useATLabel')
        dset = imageSet(cfg) 
        TrainDataSet, TestDataSet = dset.genTrainTest()

    if balance:
        prate = TrainDataSet.prate
        weights = []
        if useATLabels:
            for _, l, _ in TrainDataSet:
                weights.append(1-prate if l==1 else prate)
        else:
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
    if useATLabels:
        images, labels, _ = dataiter.next()
    else:
        images, labels = dataiter.next()
    imgGrid = torchvision.utils.make_grid(images)
    matplotlib_imshow(imgGrid, one_channel=True)
    writer.add_image('sample images', imgGrid)
    writer.add_graph(model, images)
    model.to(device)
    global globalStep
    globalStep = 0

    if useATLabels:
        trainMethod = trainClsWithAt
        lossFunc = [Clscriterion, ATMaskLoss()]
    else:
        trainMethod = trainCls
        lossFunc = Clscriterion
    try:
        for e in range(num_epochs):
            s = time.time()
            trainMethod(model, optimizer, lossFunc, TrainDataloader, TestDataloader, 
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

def testCls(model, lossFunc, testDataLoader, device, writer, step):
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

def trainClsWithAt(model, opt, lossFuncs, trainDataLoader, testDataLoader, device, writer, epoch, save_step, info_step, test_step, saveRoot, batch):
    global globalStep
    lossClsRc = LossReCollector('train cls Loss')
    lossAtRc = LossReCollector('train at loss')
    lossTotalRc = LossReCollector('train loss')
    lossAt_PosRc = LossReCollector('train at pos loss')
    lossAt_NegRc = LossReCollector('train at neg loss')

    clsoutRc = ClsReCollector() 
    atoutRc = ATReCollector()
    attargetRc = ATReCollector()
    pnRc = PNCollector()

    collectors = [lossClsRc, lossAtRc, lossTotalRc,
            lossAt_PosRc, lossAt_NegRc,
            clsoutRc, atoutRc,attargetRc, pnRc]

    clsLossFunc = lossFuncs[0]
    atLossFunc =lossFuncs[1]

    model.train()
    info_timeS = time.time()
    for i, data in enumerate(trainDataLoader):
        globalStep += 1
        inputs, labels, atmask = data[0].to(device), data[1].to(device), data[2].to(device) 
        opt.zero_grad()
        clsout, rpn = model(inputs)
        
        clsloss = clsLossFunc(clsout, labels)
        ploss, nloss = atLossFunc(rpn, atmask)
        loss =  ploss + nloss + clsloss
        loss.backward()
        opt.step()

        lossAtRc.add(ploss+nloss)
        lossAt_PosRc.add(ploss) 
        lossAt_NegRc.add(nloss)
        lossClsRc.add(clsloss)
        lossTotalRc.add(loss)

        clsoutRc.add_preds_labels(clsout, labels)
        atoutRc.add_pred(rpn)
        attargetRc.add_label(atmask)
        pnRc.add(labels)

        if globalStep % save_step == save_step-1:
            ckp = Path(f'checkpoint_{epoch}_{globalStep}.pth')
            torch.save(model.state_dict(), str(saveRoot/ckp))
        if globalStep % test_step == test_step-1:
            testClsWithAt(model, [clsLossFunc, atLossFunc], testDataLoader, device, writer, globalStep)
        if globalStep % info_step == info_step-1:
            useTime = time.time() - info_timeS
            print("epoch:{:2d}, step:{:4d} totalStep:{:6d}".format(epoch, i+1, globalStep), end='  ')
            print("{}".format(lossClsRc), end=' ')
            print("{}".format(pnRc), end=' ')
            print("{}".format(lossAtRc), end=' ')
            print(f"{useTime:5.4f}s for {info_step} step, {useTime/(info_step*batch):5.4f}s per image")
            

            atRcResult = atoutRc.getp()
            writer.add_images('train rpn pred', atRcResult, globalStep)
            atTargetRcResult = attargetRc.getl()
            writer.add_images('train rpn target', atTargetRcResult, globalStep)
            
            writer.add_scalar('training loss', lossTotalRc.get(), globalStep)
            writer.add_scalar('training cls loss', lossClsRc.get(), globalStep)
            writer.add_scalar('training At loss', lossAtRc.get(), globalStep)
            writer.add_scalar('training At p loss', lossAt_PosRc.get(), globalStep)
            writer.add_scalar('training At n loss', lossAt_NegRc.get(), globalStep)

            test_prob, gt =clsoutRc.get_result()
            writer.add_pr_curve('Training Crack PR', gt, test_prob, globalStep)
            
            for c in collectors:
                c.clear()
            torch.cuda.empty_cache()
            info_timeS = time.time()
def testClsWithAt(model, lossFuncs, testDataLoader, device, writer, step):
    print('Test start')
    s = time.time()
    model.eval()
    clsRec = ClsReCollector()
    atRec = ATReCollector()
    
    lossClsRc = LossReCollector('test cls loss')
    lossAt = LossReCollector('test at loss')
    
    clsLossFunc = lossFuncs[0]
    atLossFunc = lossFuncs[1]
    with torch.no_grad():
        for data in testDataLoader:
            inputs, labels, atmask = data[0].to(device), data[1].to(device), data[2].to(device) 
            clsout, rpn = model(inputs) 
            clsloss = clsLossFunc(clsout, labels) 
            ploss, nloss = atLossFunc(rpn, atmask)
            
            lossClsRc.add(clsloss)
            lossAt.add(ploss + nloss)
            
            clsRec.add_preds_labels(clsout, labels)
            atRec.add_pred(rpn)
            
    test_probs, gt = clsRec.get_result()
    closs = lossClsRc.get()
    atloss = lossAt.get()
    predRPN = atRec.getp()
    writer.add_scalar('Test cls loss', closs, step)
    writer.add_scalar('Test at loss', atloss, step)
    writer.add_pr_curve('Test Crack PR', gt, test_probs, step)
    writer.add_pr_curve('Test Health PR', 1-gt, 1-test_probs, step)
    writer.add_images('Test pred rpns', predRPN)
    model.train()
    useTime = time.time() - s
    print(f'Test done time:{useTime}s')

if __name__=='__main__':
    main()
