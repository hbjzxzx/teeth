import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
import torchvision
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import data
from config import config
from utils import *

configRoot = Path('configs')
configFileName = Path('resnet.yaml')
cfg = config(str(configRoot/configFileName))

# data
batchSize = 16
shuffle = True
numWorkers = 1
classes = ['healthy', 'crack']

# train
num_epochs = 50
device = torch.device('cuda:0')
outputDir = Path('modelResNet2Class')
thisRunName = Path('baseLineExp2ClsBal')
modelWeightSave = Path('weightClsBal.pth')
writer = SummaryWriter(str(outputDir/thisRunName))

# loss
criterion = nn.CrossEntropyLoss()

TrainDataSet = data.class2set(cfg, isTrain=True)
TestDataSet = data.class2set(cfg, isTrain=False)

prate = TrainDataSet.prate
weights = []
for _, l in TrainDataSet:
    weights.append(1-prate if l==1 else prate)
trainSampler = WeightedRandomSampler(weights, len(TrainDataSet), replacement=True)

TrainDataloader = DataLoader(TrainDataSet,
                                batch_size=batchSize,
                             num_workers=numWorkers,
                             sampler=trainSampler)

TestDataloader = DataLoader(TestDataSet,
                             batch_size=batchSize,
                             shuffle=shuffle,
                             num_workers=numWorkers)

resnet_model = models.resnet50(pretrained=False)
fc_features = resnet_model.fc.in_features
resnet_model.fc = nn.Linear(fc_features, 2)
optimizer = optim.SGD(resnet_model.parameters(), lr=0.0001, momentum=0.9)

dataiter = iter(TrainDataloader)
images, labels = dataiter.next()
imgGrid = torchvision.utils.make_grid(images)
matplotlib_imshow(imgGrid, one_channel=True)
writer.add_image('sample images', imgGrid)
writer.add_graph(resnet_model, images)

resnet_model.to(device)

for e in range(num_epochs):
    running_loss = 0.0
    pos = 0
    neg = 0
    for i, data in enumerate(TrainDataloader):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()

        out = resnet_model(inputs)
        loss = criterion(out, labels)
        running_loss+= loss
        pos += torch.sum(labels)
        neg += torch.sum(torch.ones_like(labels) - labels)
        loss.backward()
        optimizer.step()

        if i % 5  == 4:
            avgLoss = running_loss/4
            totalStep = e*len(TrainDataloader) + i
            print("epoch:{:2d}, step:{:5d} loss:{:.3f} posIns:{} negIns:{}".format(e+1, i+1, avgLoss, pos, neg))
            writer.add_scalar('training loss', avgLoss, totalStep)
            running_loss = 0
            pos = 0
            neg = 0

class_probs = []
class_preds = []
net = resnet_model
with torch.no_grad():
    for data in TestDataloader:
        inputs, labels = data[0].to(device), data[1].to(device)
        output = net(inputs)
        class_probs_batch = [F.softmax(el, dim=0) for el in output]
        _, class_preds_batch = torch.max(output, 1)

        class_probs.append(class_probs_batch)
        class_preds.append(class_preds_batch)

test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
test_preds = torch.cat(class_preds)

# helper function
def add_pr_curve_tensorboard(class_index, test_probs, test_preds, global_step=0):
    '''
    Takes in a "class_index" from 0 to 9 and plots the corresponding
    precision-recall curve
    '''
    tensorboard_preds = test_preds == class_index
    tensorboard_probs = test_probs[:, class_index]

    writer.add_pr_curve(classes[class_index],
                        tensorboard_preds,
                        tensorboard_probs,
                        global_step=global_step)
    writer.close()

# plot all the pr curves
for i in range(len(classes)):
    add_pr_curve_tensorboard(i, test_probs, test_preds)

writer.close()
torch.save(net.state_dict(), str(outputDir/modelWeightSave))
