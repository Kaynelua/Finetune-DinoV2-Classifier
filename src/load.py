import torch
import os
import torchvision
from torchvision import transforms, datasets

def load_dataset_from_dir(data_dir='data/'):

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224), #May not be a good idea as scale by default is 0.08 - 1.0
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                                                for x in ['train', 'test']}
    return image_datasets