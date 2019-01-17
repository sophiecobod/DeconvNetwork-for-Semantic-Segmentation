import torch
import torch.nn as nn 
import os 
import numpy as np
from load_dataset import load_dataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from model import create_model 
import time

usegpu = False

def train_model(model, dataloaders, criterion, optimizer, device, num_epochs=25):
    since = time.time()

    #val_acc_history = []

    #best_model_wts = copy.deepcopy(model.state_dict())
    #best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            #running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                if usegpu: #use gpu if available
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    outputs = torch.clamp(outputs, 0, 1)
                    loss = criterion(outputs, labels)
                    print(loss)
                    ## TODO _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                ## TODO running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            ## TODO epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            ## TODO print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            # deep copy the model
            #if phase == 'val' and epoch_acc > best_acc:
            #    best_acc = epoch_acc
            #    best_model_wts = copy.deepcopy(model.state_dict())
            # if phase == 'val':
              #  val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    #print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    # model.load_state_dict(best_model_wts)
    return model#, val_acc_history



if __name__ == "__main__":
    model = create_model()
    print("Model loaded!")

    if usegpu: #use gpu if available
        print("Using cuda")
        model.cuda()
    
    criterion = nn.BCELoss() #Binary cross entropy loss 
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

    num_epochs = 1
    train_loader, train_dataset = load_dataset("./data", "train")
    val_loader, val_dataset = load_dataset("./data", "val")
    
    print("Dataset loaded!")
    
    dataloaders = {}
    dataloaders['train'] = train_loader
    dataloaders['val'] = val_loader

    print("Training Started!")
    device = torch.device('cuda' if usegpu else 'cpu')
    model = train_model(model, dataloaders, criterion, optimizer, device, num_epochs=num_epochs)

    torch.save(model, "trained_model.pth")
    

"""
for epoch in range(num_epochs):
    print("\nEPOCH " +str(epoch+1)+" of "+str(num_epochs)+"\n")
    for i, batch in enumerate(train_loader):
        inputs, labels = batch

        optimizer.zero_grad()   
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward() #Backprop
        optimizer.step()    #Weight update
        
        writer.add_scalar('Training Loss',loss.data[0]/10, iter)
        iter=iter+1
        # Regarder structure de batch -> input
        # forward input dans le NN (output = model(input))
        
        # backward -> optimizer etc
"""
