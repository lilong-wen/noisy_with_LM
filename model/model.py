import torch
import torch.nn as nn
from torch.hub import load
import torchvision.models as models


dino_backbones = {
    'dinov2_s':{
        'name':'dinov2_vits14',
        'embedding_size':384,
        'patch_size':14
    },
    'dinov2_b':{
        'name':'dinov2_vitb14',
        'embedding_size':768,
        'patch_size':14
    },
    'dinov2_l':{
        'name':'dinov2_vitl14_reg',
        'embedding_size':1024,
        'patch_size':14
    },
    'dinov2_g':{
        'name':'dinov2_vitg14',
        'embedding_size':1536,
        'patch_size':14
    },
}


class linear_head(nn.Module):
    def __init__(self, embedding_size = 384, num_classes = 5):
        super(linear_head, self).__init__()
        self.fc = nn.Linear(embedding_size, num_classes)

    def forward(self, x):
        return self.fc(x)



class Classifier(nn.Module):
    def __init__(self, num_classes, backbone = 'dinov2_l', head = 'linear', backbones = dino_backbones):
        super(Classifier, self).__init__()
        self.heads = {
            'linear':linear_head
        }
        self.backbones = dino_backbones
        # self.backbone = load('facebookresearch/dinov2', self.backbones[backbone]['name'])
        self.backbone = load('facebookresearch/dinov2', self.backbones[backbone]['name'], pretrained=False)
        self.backbone.load_state_dict(torch.load('checkpoints/dinov2_vitl14_reg4_pretrain.pth'))
        # self.backbone = torch.load()
        self.backbone.eval()
        self.head = self.heads[head](self.backbones[backbone]['embedding_size'],num_classes)
    def forward(self, x):
        with torch.no_grad():
            x_feature = self.backbone(x)
        x = self.head(x_feature)
        return x_feature, x