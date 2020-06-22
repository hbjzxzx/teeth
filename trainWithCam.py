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
    dataConfigFileName = Path('dataconfig.yaml')
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
    class_probs = []
    class_labels = []
    for e in range(num_epochs):
        running_loss_cls = 0.0
        running_loss_at = 0.0
        running_loss = 0.0
        pos = 0
        neg = 0
        model.train()
        
        for i, data in enumerate(TrainDataloader):

            step += 1
            inputs, labels, atmask = data[0].to(device), data[1].to(device), data[2].to(device) 
            optimizer.zero_grad()
            clsOuts, atMaskOut = model(inputs)


            class_probs_batch = [F.softmax(el, dim=0)[1] for el in clsOuts.cpu()]
            class_probs.append(class_probs_batch)
            class_labels.append(labels.cpu())

            Clsloss = Clscriterion(clsOuts, labels)
            atMaskLoss = ATcriterion(atMaskOut, atmask)
            loss = Clsloss + atMaskLoss

            running_loss_cls+= Clsloss
            running_loss_at += atMaskLoss
            running_loss += loss

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
                totalStep = e*len(TrainDataloader) + i

                test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
                gt = torch.cat(class_labels)
                
                writer.add_pr_curve('healthy train', 1-gt , 1 - test_probs, global_step=step)
                writer.add_pr_curve('crack train', gt , test_probs, global_step=step)
                class_probs.clear()
                class_labels.clear()

                print("epoch:{:2d}, step:{:4d} TotalStep:{:4d} loss:{:.3f} ClsLoss:{:.3f} AtLoss:{:.3f} posIns:{} negIns:{}".format(e+1, i+1, step, avgLoss, avgClsLoss, avgAtLoss, pos, neg))
                writer.add_scalar('training loss', avgLoss, step)
                writer.add_scalar('training cls loss', avgClsLoss, step)
                writer.add_scalar('training At loss', avgAtLoss, step)
                running_loss = 0
                running_loss_cls = 0
                running_loss_at = 0
                pos = 0
                neg = 0
        model.eval()
        test(model, TestDataloader, device, writer, step)
    torch.save(model.state_dict(), str(saveRoot/Path('model_final.pth')))

def test(model, testdata, device,writer,  step):
    class_probs = []
    class_labels = []
    pred_At = []
    #inImages = []
    net = model
    with torch.no_grad():
        for data in testdata:
            inputs, labels = data[0].to(device), data[1].to(device)
            outcls, outmask = net(inputs)
            class_probs_batch = [F.softmax(el, dim=0)[1] for el in outcls.cpu()]

            class_probs.append(class_probs_batch)
            class_labels.append(labels.cpu())
            pred_At.append(outmask.cpu())
            #inImages.append(inputs)

    test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
    gt = torch.cat(class_labels)
    #inImages = torch.cat(inImages)

    pred_At = torch.cat(pred_At)
    pred_At.unsqueeze_(dim=1).repeat(1,3,1,1)

    writer.add_images('pred Attention', pred_At, global_step=step)
    #writer.add_images('images', inImages)

    add_pr_curve_tensorboard(0, 1-gt , 1 - test_probs, writer, global_step=step)
    add_pr_curve_tensorboard(1, gt , test_probs,writer, global_step=step)


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
