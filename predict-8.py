# Import Python Modules
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
    
    """
        Command Line Arguments with the use of the arparse module
    """
    parser = argparse.ArgumentParser(description='Input your arguments')

    parser.add_argument('image_path', action='store', type=str, 
                    default='flowers/test/1/image_06743.jpg', 
                    help='path to the image to be recognised')
    
    parser.add_argument('--save_dir', type=str, action='store', default='checkpoint.pth',
                    help='directory to get the saved checkpoint from')
    
    parser.add_argument('--arch', type=str, default="vgg16",
                    help="provide model architecture: vgg16, resnet18, alexnet")
    
    parser.add_argument('--gpu', default=True,
                    help='use GPU or CPU to train model: True = GPU, False = CPU')
    
    parser.add_argument('--top_k', default=5, type=int, 
                    help="Number of top predictions/ probabilities")                

    
    parser.add_argument('--cat_to_name', action='store', default = 'cat_to_name.json',
                    help='Enter path to image.')

    args = parser.parse_args() 
                        
    return args
                        
def load_checkpoint(save_dir):
    """
       Loads a saved checkpoint.pth file
    """
    
    #checkpoint = torch.load(save_dir)
    checkpoint=torch.load(save_dir, map_location=lambda storage, loc: storage)
    arch = checkpoint['arch']
    hidden_units = checkpoint['hidden_units']
 
    model = getattr(models,arch)(pretrained=True)
    in_features = model.classifier[0].in_features
    
    for param in model.parameters():
        param.requires_grad=False
        
    model.class_to_idx = checkpoint['class_to_idx']
    
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(in_features, hidden_units, bias=True)),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout(p=0.5)),
                          ('fc2', nn.Linear(hidden_units, 102, bias=True)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])


    return model

    
def process_image(image_path):
    """
    Function to load and pre-process test image
    
    Scales, crops, and normalizes a PIL image for a PyTorch model,returns a Numpy array
    """
                        
    # Load Image
    img = Image.open(image_path)
    
    # Get the dimensions of the image
    width, height = img.size
    
    # Resize with the shortest size being 256px
    img = img.resize((256, int(256*(height/width))) if width < height else (int(256*(width/height)), 256))
    
    #Get the dimensions of the new image size
    width, height = img.size
    
    # Set the coordinates to do a center crop of 224 x 224
    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2
    img = img.crop((left, top, right, bottom))
    
    # Turn image into numpy array
    img = np.array(img)
    
    # Make the color channel dimension first instead of last
    img = img.transpose((2, 0, 1))
    
    # Make all values between 0 and 1
    img = img/255
    
    # Normalize based on the preset mean and standard deviation
    img[0] = (img[0] - 0.485)/0.229
    img[1] = (img[1] - 0.456)/0.224
    img[2] = (img[2] - 0.406)/0.225
    
    
    # Turn into a torch tensor
    image = torch.from_numpy(img)
    image = image.float()
    
    return image



def predict(image_path, model, top_k,cat_to_name, device):
    
    """
    Predict the class (or classes) of an image using a trained deep learning model.
    """
    
    #Run on gpu, if not available - then cpu.
 
    if torch.cuda.is_available():
        model.cuda()
    else:
        model.to(device)
    
    #Move model into evaluation mode
    model.eval()
    
    #Load and process the image 
    image = Image.open(image_path)
    imgage = process_image(image_path)
    imgage = imgage.unsqueeze(0)
    imgage = imgage.float()
    
    #Run model 
    with torch.no_grad():
        output = model.forward(imgage.to(device))
    
    #Convert softmax output to probabilities     
    probability = F.softmax(output.data,dim=1)
    
    #Find top-k probabilities and indices 
    top_probability = np.array(probability.topk(k=top_k)[0][0])
    #Invert dictionary
    index_to_class = {val: key for key, val in model.class_to_idx.items()} 
    
    #Find the class using the indices
    top_classes = [np.int(index_to_class[each]) for each in np.array(probability.topk(top_k)[1][0])] 
    
    #Map the class name with top-k classes
    names = []
    
    for classes in top_classes:
        names.append(cat_to_name[classes])
            
    #rfunction returns top probability and top classes
    return top_probability, top_classes



def main():
    
    args = get_args()
    
    with open(args.cat_to_name, 'r') as f:
            cat_to_name = json.load(f)
     
    #Load pre-trained model from checkpoint
    model = load_checkpoint(args.save_dir)
    
    
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu==True else 'cpu') 
    
    print('Device is: ', device)

    #Prediction
    top_probability, top_classes = predict(args.image_path, model, args.top_k, cat_to_name, device)
    
    print(top_probability)
    print(top_classes)
    
    ## Print name of predicted flower with highest probability - print(f"This flower is most likely to be a: '{names[0]}' with a probability of {round(probs[0]*100,4)}% ")

if __name__ == "__main__":
    main()   