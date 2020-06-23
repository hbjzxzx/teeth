import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from pathlib import Path
from inception import Inception3
from AtNets import ATRestNet, ATClass, ATClassFPN
regNets = {}

def reg(text):
    def inner(func):
        if text not in regNets.keys():
            regNets[text] =  func
        else:
            raise Exception(f"register twice func{func.__name__}")
        return func
    return inner

@reg('resnet50')
def getRestNet(**kwargs):
    config = kwargs['config']
    netName = config['train']['netname']
    pretrain = config['train']['pre_train']
    resume = config['train']['resume']
    resume_path = Path(config['train']['resume_path'])
    assert not ( pretrain and resume), 'pre_trian or resume can not turn on same simultaneously'
    pretrainPath = Path(f"pretrain/{netName}.pth")
    resnet_model = models.resnet50(pretrained=False)
    if pretrain:
        resnet_model.load_state_dict(torch.load(str(pretrainPath)))
        print(f"using pretrain weight:{str(pretrainPath)}")
        fc_features = resnet_model.fc.in_features
        resnet_model.fc = nn.Linear(fc_features, 2)
    elif resume:
        print(f'resume the train from weight:{str(resume_path)}')
        fc_features = resnet_model.fc.in_features
        resnet_model.fc = nn.Linear(fc_features, 2)
        resnet_model.load_state_dict(torch.load(str(resume_path)))
    else:
        fc_features = resnet_model.fc.in_features
        resnet_model.fc = nn.Linear(fc_features, 2)

    return resnet_model


@reg('inceptionv3')
def getIncep(**kwargs):
    config = kwargs['config']
    netName = config['train']['netname']
    pretrain = config['train']['pre_train']
    resume = config['train']['resume']
    resume_path = Path(config['train']['resume_path'])

    assert not ( pretrain and resume), 'pre_trian or resume can not turn on same simultaneously'
    pretrainPath = Path(f"pretrain/{netName}.pth")

    if resume:
        assert resume_path.exists(), f'resume checkpoint {str(resume_path)} not exist! '

    print('building inception model')
    model = Inception3(transform_input=True)
    if pretrain:
        print('loading weights from pretraind pth file')
        model.load_state_dict(torch.load(str(pretrainPath)))
        model.fc = nn.Linear(2048, 2)
    elif resume:
        model.load_state_dict(torch.load(str(resume_path)))
    return model


@reg('ATresnet')
def getATResnet(**kwargs):
    config = kwargs['config']
    netName = config['train']['netname']
    pretrain = config['train']['pre_train']
    resume = config['train']['resume']
    resume_path = Path(config['train']['resume_path'])
    if resume:
        assert resume_path.exists(), f'resume file {str(resume_path)} not exist'
    assert not (pretrain and resume), 'pre_trian or resume can not turn on same simultaneously'
    pretrainPath = Path(f"pretrain/{netName}.pth") if pretrain else None


    net = ATRestNet(pretrainPath)
    if resume:
        print(f'resume the train from weight:{str(resume_path)}')
        net.load_state_dict(torch.load(str(resume_path)))

    return net


@reg('ATClsResnet')
def getARClsResnet(**kwargs):
    return ATClass()

@reg('ATClsFPNResnet')
def getATCLSFPNnet(**kwargs):
    return ATClassFPN()