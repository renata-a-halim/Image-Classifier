#Import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import argparse
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from collections import OrderedDict

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from PIL import Image
import json



def get_args():
    parser = argparse.ArgumentParser(description='Input your arguments.')

    parser.add_argument('data_dir', type=str,
                    help='data directory containing training and testing dataset') 
    
    parser.add_argument('--arch', type=str, default="vgg16",
                    help='provide model architecture from torchvision.models e.g. vgg16, alexnet etc.')
                        
    parser.add_argument('--hidden_units',type=list, default=4096,
                    help='customise the classifier network by providing list of hidden layers, default is 4096')
    
    parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='provide learning rate of the optimizer, default is 0.001')
    
    parser.add_argument('--dropout', action = 'store', type=float, default = 0.05,
                    help ='provide dropout for training the model, default is 0.05.')
    
    parser.add_argument('--epochs', type=int, default=1,
                    help='provide number of epochs to train the model, default is 1')
    
    parser.add_argument('--output', type=int, default=102,
                    help='enter output size')
    
    parser.add_argument('--save_dir', type=str, default='checkpoint.pth',
                    help='directory for saving the checkpoint file')
    
    parser.add_argument('--gpu', default=True,
                    help='use GPU or CPU to train model: True = GPU, False = CPU')
    
    args = parser.parse_args() 
                        
    return args
    

def load_data():
                        
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_transforms = transforms.Compose([transforms.RandomRotation(20),
                                                               transforms.RandomResizedCrop(224),
                                                               transforms.RandomHorizontalFlip(),
                                                               transforms.ToTensor(),
                                                               transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                                                    std=[0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(256),
                                                               transforms.CenterCrop(224),
                                                               transforms.ToTensor(),
                                                               transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                                                    std=[0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                                              transforms.CenterCrop(224),
                                                              transforms.ToTensor(),
                                                              transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                                                   std=[0.229, 0.224, 0.225])])


    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)
    
    return train_dataset, valid_dataset, test_dataset, train_loader, valid_loader, test_loader



def build_model(arch, hidden_units, learning_rate):
    model =  getattr(models,arch)(pretrained=True)
    in_features = model.classifier[0].in_features

    for param in model.parameters():
        param.requires_grad = False
        
    
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(in_features, hidden_units, bias=True)),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout(p=0.5)),
                          ('fc2', nn.Linear(hidden_units, 102, bias=True)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    model.classifier = classifier
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device
    
    model.to(device);
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    return model, criterion, optimizer


def validation(model, valid_loader, criterion, device='cuda'):
    valid_loss = 0
    accuracy = 0
    
    for ii, (inputs, labels) in enumerate(valid_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        output = model.forward(inputs)
        valid_loss += criterion(output, labels).item()
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
        
    
    return valid_loss, accuracy
                        
    
def train_model(model, train_dataset, train_loader, valid_loader, test_loader, optimizer, criterion, epochs, device='cuda'):
    model.train()
    
    print("Training process starts .....\n")
   
    
     # Define deep learning method
    print_every = 30 # Prints every 30 images out of batch of 64 images
    steps = 0

    for e in range(epochs):
        running_loss = 0
        
        for ii, (inputs, labels) in enumerate(train_loader):
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if steps % print_every == 0:
                model.eval()
                with torch.no_grad():
                    valid_loss, accuracy = validation(model, valid_loader, criterion)
                print("Epoch: {}/{} | ".format(e+1, epochs),
                  "Training Loss: {:.4f} | ".format(running_loss/print_every),
                  "Validation Loss: {:.4f} | ".format(valid_loss/len(valid_loader)),
                  "Validation Accuracy: {:.4f}".format(accuracy/len(valid_loader)))
                running_loss = 0
                model.train()
    
    model.class_to_idx = train_dataset.class_to_idx
    
    print("\nTraining process has ended.")
    print("Test is starting...")


#test the accuracy of the model on test dataset
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        model.eval()
    
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()
    
    
    
    model.class_to_idx = train_dataset.class_to_idx
                                           
    
    print('Test accuracy ({0:d} validation images): {1:.1%}'.format(test_total, test_correct / test_total))
    
    return model

def save_checkpoint(model, save_dir, optimizer, epochs, arch, hidden_units):
    checkpoint = {
        'arch': arch,
        'output_size': 102,
        'hidden_units': hidden_units,
        'epoch': epochs,
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx,
        'optimizer': optimizer.state_dict()}

    #torch.save(checkpoint, save_dir + '/checkpoint.pth')
    torch.save(checkpoint, save_dir)

    
    return checkpoint
    
    

def main():
    #load the category names                    
    with open('cat_to_name.json', 'r') as f:
        flower_names = json.load(f)
                        
    args = get_args()# get user arguments
    
   # print arguments

    print('Data directory: ' + args.data_dir)
    print('Architecture: ' + args.arch)
    print('Hidden units: ' + str(args.hidden_units))
    print('Epochs:' +str(args.epochs))
    print('Learning rate:' +str(args.learning_rate))
    print('Save directory: ' + args.save_dir)
    
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu==True else 'cpu') 
    print("Device:", device)
    
    #load the datasets, transforms and loaders
    train_dataset, valid_dataset, test_dataset, train_loader, valid_loader, test_loader = load_data()
 
    #get the model                    
    model, criterion, optimizer = build_model(args.arch, args.hidden_units, args.learning_rate)
    
    #train the model
    
    model = train_model(model, train_dataset, train_loader, valid_loader, test_loader, optimizer, criterion, args.epochs, device='cuda')
    
    
    #save the model
    checkpoint = save_checkpoint(model, args.save_dir, optimizer, args.epochs, args.arch, args.hidden_units)
    
    print('Checkpoint has been saved')
    

     
if __name__ == '__main__':
    main()                       
                        
                        
    
    