import  torch
import torch.nn as nn
import  torch.nn.functional as F
from torchvision import models
from pathlib import  Path


class ATRestNet(nn.Module):
    def __init__(self, pretrainedPath:Path=None, resumePath:Path=None):
        super(ATRestNet, self).__init__()
        hasPreTrained = pretrainedPath is not None

        if hasPreTrained:
           assert pretrainedPath.exists(), f'preTrained Path {str(pretrainedPath)} not exist'

        resnet = models.resnet50(pretrained=False)
        if hasPreTrained:
            resnet.load_state_dict(torch.load(str(pretrainedPath)))

        self.resnet_layer = nn.Sequential(*(list(resnet.children())[:-2]))

        #input images batch*3*512*512 resnet out : batch*2048*16*16
        self.atConv1 = nn.Conv2d(2048, 256, kernel_size=3, stride=1, padding=1)
        self.re = nn.ReLU()
        self.atConv2 = nn.Conv2d(256, 2, kernel_size=1)

        
        if resumePath:
            print(f'resume model from {str(resumePath)}')
            try:
                self.load_state_dict(torch.load(str(resumePath)))
            except Exception as e:
                print(f'{str(e)}')

    def forward(self, x):
        bbf = self.resnet_layer(x)
        rpn = self.atConv1(bbf)
        rpn = F.relu(rpn)
        rpn = self.atConv2(rpn)
        return rpn, bbf 
        

class ATClass(nn.Module):
    def __init__(self, rpnTh=0.1):
        super(ATClass, self).__init__()
        ATNetResumePath = Path('pretrain/ATmodel_final.pth')
        self.AT_FNet = ATRestNet(resumePath=ATNetResumePath)
        self.clsConv1 = nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1)
        self.globalAvgPool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 2)
        self.rpnTh = 0.1
    
    def forward(self, x):
        rpn, bbf = self.AT_FNet(x)
        rpnAt = torch.softmax(rpn, dim=1)[:,1]
        rpnAt = torch.ge(rpnAt, self.rpnTh) 
        atFeature = torch.mul(bbf, rpnAt.unsqueeze(1))
        atFeature = self.clsConv1(atFeature)
        atFeature = F.relu(atFeature)
        atFeature = self.globalAvgPool(atFeature)
        atFeature = atFeature.flatten(1)
        clsout = self.fc(atFeature)
        return clsout, rpn


class ATClassFPN(nn.Module):
    def __init__(self, rpnTh=0.1):
        super(ATClassFPN, self).__init__()
        ATNetResumePath = Path('pretrain/ATmodel_final.pth')
        self.AT_FNet = ATRestNet(resumePath=ATNetResumePath)

        childNets = list(self.AT_FNet.children())
        resnetParts = list(childNets[0].children())

        self.ATheads = nn.Sequential(*(childNets[1:]))  

        self.resHead = nn.Sequential(*(resnetParts[:-4]))
        self.resC2Layer = resnetParts[-4]
        self.resC3Layer = resnetParts[-3]
        self.resC4Layer = resnetParts[-2]
        self.resC5Layer = resnetParts[-1]

        self.C2DownChannelLayer = nn.Conv2d(256, 256, kernel_size=1, stride=1)
        self.C3DownChannelLayer = nn.Conv2d(512, 256, kernel_size=1 , stride=1)
        self.C4DownChannelLayer = nn.Conv2d(1024, 256, kernel_size=1, stride=1)
        self.C5DownChannelLayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1)


        self.smoothC2Layer = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smoothC3Layer = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smoothC4Layer = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        #self.clsConv1 = nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1)
        self.globalAvgPool = nn.AdaptiveAvgPool2d((2,2))
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(256*4*4, 2)
        self.rpnTh = 0.1
    

    def forward(self, x):
        x = self.resHead(x)
        c2 = self.resC2Layer(x)
        c3 = self.resC3Layer(c2)
        c4 = self.resC4Layer(c3)
        c5 = self.resC5Layer(c4)
        rps = self.ATheads(c5)
        
        rpsSoft = torch.softmax(rps, dim=1)[:,1,:,:]

        #rpsSoft = torch.ge(rpsSoft, self.rpnTh)
        #rpsSoft = rpsSoft.float()
        #rpsUse = rpsSoft.unsqueeze(1)

        rpsUse = rpsSoft.unsqueeze(1)

        #c5 = self._upsample_add_AT(rpsUse, c5)
        #c4 = self._upsample_add_AT(rpsUse, c4)
        #c3 = self._upsample_add_AT(rpsUse, c3)
        #c2 = self._upsample_add_AT(rpsUse, c2)

        p5 = self.C5DownChannelLayer(c5)
        p4 = self._upsample_add(p5, self.C4DownChannelLayer(c4))
        p3 = self._upsample_add(p4, self.C3DownChannelLayer(c3))        
        p2 = self._upsample_add(p3, self.C2DownChannelLayer(c2))

        p4 = self.smoothC4Layer(p4)
        p3 = self.smoothC3Layer(p3)
        p2 = self.smoothC2Layer(p2)


        p5At = self._upsample_add_AT(rpsUse, p5) 
        p4At = self._upsample_add_AT(rpsUse, p4)
        p3At = self._upsample_add_AT(rpsUse, p3)
        p2At = self._upsample_add_AT(rpsUse, p2)
        
        p5Pool = self.globalAvgPool(p5At).flatten(1)
        p4Pool = self.globalAvgPool(p4At).flatten(1)
        p3Pool = self.globalAvgPool(p3At).flatten(1)
        p2Pool = self.globalAvgPool(p2At).flatten(1)
        
        #p5Pool = self.globalAvgPool(p5).flatten(1)
        #p4Pool = self.globalAvgPool(p4).flatten(1)
        #p3Pool = self.globalAvgPool(p3).flatten(1)
        #p2Pool = self.globalAvgPool(p2).flatten(1)
        featureCls = torch.cat([p5Pool, p4Pool, p3Pool, p2Pool], dim=1)
         
        featureCls = self.drop(featureCls)
        featureCls = self.fc(featureCls)
        return featureCls, rps


    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') + y

    
    def _upsample_add_AT(self, At, destFeature):
        _,_,H,W = destFeature.size()
        return torch.mul(F.upsample(At, size=(H,W), mode='bilinear'), destFeature)

class FocalLoss(nn.Module):
    def __init__(self, posWeight):
        super(FocalLoss, self).__init__() 
        self.gamma =5
        self.posBias = posWeight 
        print(f'using focal loss gamms:{self.gamma} PosWeight:{self.posBias}')

    
    def forward(self, input, target):
        input = torch.softmax(input, dim=1)
        target = target.unsqueeze(1)
        target = torch.cat([1-target, target], dim=1)
        input = input * target
        input = input.sum(dim=1) 
        loss = (1-input)**self.gamma*-torch.log(input)
        clsWeight =self.posBias*(target[:,1]) + (1-self.posBias)*(target[:,0])
        loss = loss*clsWeight
        return loss.mean()

class ClsWeightLoss(nn.Module):
    def __init__(self, posWeight):
        super(ClsWeightLoss, self).__init__()
        self.posWeight = posWeight
        print(f'using ClsWeight pos Weight is {self.posWeight}')


    def forward(self, input, target):
        input = torch.softmax(input, dim=1)
        target = target.unsqueeze(1)
        target = torch.cat([1-target, target], dim=1)
        loss = -torch.log(input) * target
        loss = loss.sum(dim=1)
        loss = torch.where(target[:, 1] == 1, loss * self.posWeight, loss * (1-self.posWeight)) 
        return loss.mean() 

class ATMaskLoss(nn.Module):
    def forward(self, input, target):
        input = F.softmax(input, dim=1)
        input = torch.clamp(input, 1e-4, 1-1e-4)
        #posLoss = -1*target*torch.log(input)
        #posNum = torch.clamp((target==1).sum(), 1)
        #posLoss = posLoss/posNum

        #negLoss = -1*(1 - target)*torch.log(1 - input)
        #negNum = torch.clamp((target==0).sum(), 1)
        #negLoss = negLoss/negNum

        #loss =  posLoss + negLoss
        #loss = torch.where(target != -1, loss, torch.zeros_like(loss))
        #loss = torch.clamp(loss, -10 ,10)
        posNum = torch.clamp((target==1).sum(), 1)
        negNum = torch.clamp((target==0).sum(), 1)
        posLoss = -torch.log(input[:,1,:,:]) * target / posNum
        posLoss = torch.where(target==1, posLoss, torch.zeros_like(posLoss))
        
        negLoss = -torch.log(input[:,0,:,:]) * (1-target) / negNum
        negLoss = torch.where(target==0, negLoss, torch.zeros_like(negLoss))
        return posLoss.mean(), negLoss.mean()

