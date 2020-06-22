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
from data import class2setWithATMask
from utils import *
from AtNets import ATMaskLoss
classes = ['healthy', 'crack']

def main():
    net = 'ATresnet'
    #net = 'resnet'
    #net = 'inceptionv3'

    configRoot = Path('configs')
    dataConfigFileName = Path('dataconfig2.yaml')
    netConfigFileName = Path(f'{net}.yaml')
    cfg = config(str(configRoot/dataConfigFileName))
    cfg.mergeWith(str(configRoot/netConfigFileName))

    # data
    batchSize = cfg['train']['batch_size']
    shuffle = cfg['train']['shuffle']
    numWorkers = cfg['train']['num_worker']
    balance = cfg['train']['balance']

    # train
    netname = cfg['train']['netname']
    num_epochs = cfg['train']['epoch']
    device = cfg['train']['device']
    device = torch.device(device)
    outputDir = Path(cfg['train']['output_dir'])
    netoutdir = Path(netname)
    thisRunName = Path(cfg['train']['session_dir'])

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

    # loss
    Clscriterion = nn.CrossEntropyLoss()
    ATcriterion = ATMaskLoss()

    TrainDataSet = class2setWithATMask(cfg, isTrain=True)
    TestDataSet = class2setWithATMask(cfg, isTrain=False)
    print(f'TrainDataSet positive rate {TrainDataSet.prate}')
    print(f'TestDataSet positive rate {TestDataSet.prate}')

    if balance:
        prate = TrainDataSet.prate
        weights = []
        for _, l, _ in TrainDataSet:
            weights.append(1-prate if l==1 else prate)
        trainSampler = WeightedRandomSampler(weights, len(TrainDataSet), replacement=True)

        TrainDataloader = DataLoader(TrainDataSet,
                                    batch_size=batchSize,
                                    num_workers=numWorkers,
                                        sampler=trainSampler)
        print("using balance data")
    else:
        TrainDataloader  = DataLoader(TrainDataSet,
                                    batch_size=batchSize,
                                    shuffle=shuffle,
                                    num_workers=numWorkers)

    TestDataloader = DataLoader(TestDataSet,
                                    batch_size=batchSize,
                                    shuffle=False,
                                    num_workers=numWorkers)

    model = regNets[net](config=cfg)
    optimizer = optim.SGD(model.parameters(), lr=1e-5, momentum=0.9)
    dataiter = iter(TrainDataloader)
    images, labels, masks = dataiter.next()
    imgGrid = torchvision.utils.make_grid(images)
    MaskGrid = torchvision.utils.make_grid(masks.unsqueeze(1).repeat(1,3,1,1))

    writer.add_image('sample images', imgGrid)
    writer.add_image('sample image mask', MaskGrid)
    writer.add_graph(model, images)

    model.to(device)
    step = 1
    pred_at = []
    label_at = []
    for e in range(num_epochs):
        running_loss_cls = 0.0
        running_loss_at = 0.0
        running_loss = 0.0
        running_at_ploss = 0.0
        running_at_nloss = 0.0
        pos = 0
        neg = 0
        model.train()
        
        for i, data in enumerate(TrainDataloader):

            step += 1
            inputs, labels, atmask = data[0].to(device), data[1].to(device), data[2].to(device) 
            optimizer.zero_grad()
            rpn = model(inputs)

            ploss, nloss = ATcriterion(rpn, atmask)
            loss =  ploss + nloss
            loss = loss

            running_loss_at += loss
            running_loss += loss
            running_at_ploss += ploss
            running_at_nloss += nloss

            pos += torch.sum(labels)
            neg += torch.sum(torch.ones_like(labels) - labels)
            loss.backward()
            optimizer.step()
            if step % save_step == 0:
                ckp = Path(f'checkpoint_{e+1}_{i+1}_{step+1}.pth')
                torch.save(model.state_dict(), str(saveRoot/ckp))
            if i % 5  == 4:
                avgLoss = running_loss/5
                avgClsLoss = running_loss_cls/5
                avgAtLoss = running_loss_at/5
                avgAtPLoss = running_at_ploss/5
                avgAtNLoss = running_at_nloss/5

                totalStep = e*len(TrainDataloader) + i
                rpn = rpn.cpu()[:2]
                rpn = torch.argmax(rpn, dim=1)
                rpn = rpn.unsqueeze(1).repeat(1,3,1,1)
                writer.add_images('train rpn pred', rpn)
                atmask = atmask.cpu()[:2]
                atmask = atmask.unsqueeze(1).repeat(1,3,1,1)
                writer.add_images('train rpn target', atmask) 

                print("epoch:{:2d}, step:{:4d} TotalStep:{:4d} loss:{:.3f} ClsLoss:{:.3f} AtLoss:{:.3f} posIns:{} negIns:{}".format(e+1, i+1, step, avgLoss, avgClsLoss, avgAtLoss, pos, neg))
                writer.add_scalar('training loss', avgLoss, step)
                writer.add_scalar('training cls loss', avgClsLoss, step)
                writer.add_scalar('training At loss', avgAtLoss, step)
                writer.add_scalar('training At p loss', avgAtPLoss, step)
                writer.add_scalar('training At n loss', avgAtNLoss, step)
                running_loss = 0
                running_loss_cls = 0
                running_loss_at = 0
                running_at_ploss = 0
                running_at_nloss = 0
                pos = 0
                neg = 0
        model.eval()
        test(model, TestDataloader, device, writer, step)
    torch.save(model.state_dict(), str(saveRoot/Path('model_final.pth')))

def test(model, testdata, device,writer,  step):
    pred_At = []
    net = model
    with torch.no_grad():
        for data in testdata:
            inputs, labels = data[0].to(device), data[1].to(device)
            outmask = net(inputs)
            outmask = outmask.cpu()[:1]
            outmask = torch.argmax(outmask, dim=1)
            outmask = outmask.unsqueeze(1).repeat(1,3,1,1)
            pred_At.append(outmask)
            #inImages.append(inputs)
    
    pred_At = torch.cat(pred_At, dim=0)
    writer.add_images('pred Attention', pred_At, global_step=step)



# helper function
def add_pr_curve_tensorboard(class_index, gt, pred_pro, writer, global_step=0):
    '''
    Takes in a "class_index" from 0 to 9 and plots the corresponding
    precision-recall curve
    '''
    writer.add_pr_curve(classes[class_index],
                        gt,
                        pred_pro,
                        global_step=global_step)

if __name__=='__main__':
    main()
