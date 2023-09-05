import torch
import torch.nn as nn
import os
import torchvision
from torchvision import transforms, datasets
from dinov2.models.vision_transformer import vit_large

#Load dinov2_vitl14 from offline weights
dinov2_vitl14 = vit_large(patch_size=14,
                img_size=526,
                init_values=1.0,
                block_chunks=0 )
dinov2_vitl14.load_state_dict(torch.load('/home/dinov2/dinov2_vitl14_pretrain.pth'))
#dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14') # load model from internet torch hub repo

class DinoVisionTransformerClassifier(nn.Module):
    def __init__(self, num_classes = 4):
        super(DinoVisionTransformerClassifier, self).__init__()
        self.transformer = dinov2_vitl14
        self.num_classes = num_classes
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_classes)
        )
    
    def forward(self, x):
        x = self.transformer(x)
        x = self.transformer.norm(x)
        x = self.classifier(x)
        return x