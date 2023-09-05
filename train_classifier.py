import torch
import torch.nn as nn
import torch. optim as optim
from torch.optim import lr_scheduler

import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import os
import random
from src.load import load_dataset_from_dir
from src.model import DinoVisionTransformerClassifier



#Load dataset from dir with relevant transforms and folder name as class name
data_dir = 'data/places365'
image_datasets = load_dataset_from_dir(data_dir)
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16,shuffle=True, num_workers=2)
											for x in ['train', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#train_features, train_labels = next(iter(dataloaders['train']))


#Load DinoV2 Model
dinov2l14_pretrained_model = DinoVisionTransformerClassifier(num_classes = 4)
model = dinov2l14_pretrained_model.to(device)

#Define Loss and Optimizers
criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=0.000001)


#Number of training epochs

for epoch in range(10):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(dataloaders["train"], 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs.to(device))
        loss = criterion(outputs, labels.to(device))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 50 == 49:    # print every 50 batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 50:.4f}')
            running_loss = 0.0

print("Training completed. Initiating Evaluation")