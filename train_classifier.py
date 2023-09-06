import torch
import torch.nn as nn
import torch. optim as optim
from torch.optim import lr_scheduler

import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import os
import random
from tqdm import tqdm
from src.load import load_dataset_from_dir
from src.model import DinoVisionTransformerClassifier



#Load dataset from dir with relevant transforms and folder name as class name
data_dir = 'data/places365'
image_datasets = load_dataset_from_dir(data_dir)
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16,shuffle=True, num_workers=2)
											for x in ['train', 'val', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#train_features, train_labels = next(iter(dataloaders['train']))


#Load DinoV2 Model
dinov2l14_pretrained_model = DinoVisionTransformerClassifier(num_classes = len(class_names))
model = dinov2l14_pretrained_model.to(device)

#Define Loss and Optimizers
criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=0.000001)


min_val_loss = np.Inf  # Define initial min val loss at infinity
epochs_with_no_improvement = 0  # Counter for early stopping

#Number of training epochs
NUM_EPOCHS = 5
for epoch in range(NUM_EPOCHS):  # loop over the dataset multiple times
    print('Epoch {}/{}'.format(epoch, NUM_EPOCHS-1))
    print('-' * 10)

    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            model.eval()  # Set model to evaluate mode

        # Initialize metrics for this phase
        running_loss = 0.0  # Accumulate losses over the epoch
        correct = 0  # Count correct predictions
        total = 0  # Count total predictions
        
        with tqdm(total=len(dataloaders[phase]), unit='batch') as p:
            #Iterate over mini-batches
            for inputs,labels in dataloaders[phase] :
                # get the inputs as a list of [inputs, labels]
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize pass only in training phase
                with torch.set_grad_enabled(phase == 'train'): 
                    outputs = model(inputs.to(device))
                    _, preds = torch.max(outputs, 1)  # Get the class with the highest probability
                    loss = criterion(outputs, labels.to(device))
                    
                    if phase == 'train':
                        loss.backward() # Calculate gradients based on the loss
                        optimizer.step() # Update model parameters based on the current gradient

		        # print statistics
                running_loss += loss.item() * inputs.size(0) #Multiply average loss by batch size
                total += labels.size(0)
                correct += (preds == labels.to(device)).sum().item() 
                p.set_postfix({'loss': loss.item(), 'accuracy': 100 * correct / total})
                p.update(1)

            # Calculate loss and accuracy for this epoch
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = 100 * correct / total

            print('{} Loss: {:.4f} Acc: {:.2f}%'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val':
                if epoch_loss < min_val_loss:
                    print(f'Validation Loss Decreased({min_val_loss:.5f}--->{epoch_loss:.5f}) \t Saving The Model')
                    min_val_loss = epoch_loss  # Update minimum validation loss
                    torch.save(model.state_dict(), 'weights/classification_model.pt')  # Save the current model weights
                    epochs_with_no_improvement = 0  # Reset epochs since last improvement
                else:
                    epochs_with_no_improvement += 1
                    # TODO: Implement early stopping

print("Training completed.")