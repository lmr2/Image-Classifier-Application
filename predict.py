# Imports here

import time
import os
import json
import torch
from torch import nn, optim
from torchvision import transforms, datasets, models
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict
from PIL import Image
import argparse

# TODO: Write a function that loads a checkpoint and rebuilds the model

def model_type(model_name):
    model = None                   
    if model_name == 'vgg11':
        model = models.vgg11(pretrained=True)
    if model_name == 'vgg13':
        model = models.vgg13(pretrained=True)
    if model_name == 'vgg16':
        model = models.vgg16(pretrained=True)
# TODO: Build and train your network
# First freeze parameters so we don't backpropegate through them
    for param in model.parameters():
        param.requires_grad = False
    return model

# Set device as cpu or gpu  

def device_to_use(device):
    if device == "cpu":
        torch.device("cpu")
        print("Device used is {}.".format(device))
    if device == "cuda":
        torch.device("cuda")
        print("Device used is {}.".format(device))
    else:
        raise ValueError("Choose device as either cpu or cuda")
        
# TODO: Write a function that loads a checkpoint and rebuilds the model

# criterion = nn.NLLLoss()
# optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
# optimizer.load_state_dict(checkpoint['optimizer'])

def load_checkpoint(filepath):

    checkpoint = torch.load(filepath)  
    model_name = checkpoint['architecture']
    model = model_type(model_name)
    
    model.load_state_dict(checkpoint['state_dict'], strict=False) 
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
        
    return model

# TODO: Process a PIL image for use in a PyTorch model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    pil_image = Image.open(image)
    transform = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
    tensor_image = transform(pil_image)
    return tensor_image

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    device = 'cpu'
    model.to(device)
    processed_image = process_image(image_path)
    processed_image = processed_image.unsqueeze_(0)
    processed_image = processed_image.float()
    processed_image = processed_image.to(device)
    model.eval()
    
    with torch.no_grad():
        output = model(processed_image)
        probs = torch.exp(output).data.numpy()[0]
        index = np.argsort(probs)[-topk:][::-1]
        index_class_mapping = { i : j for j, i in model.class_to_idx.items()}
        classes = [index_class_mapping[x] for x in index]
        flower_names = []
        for i in classes:
            flower_names.append(cat_to_name[str(i)])
        probs = probs[index]
    
    return probs, flower_names


def print_flower_and_classes(image_file, model):
# TODO: Display an image along with the top 5 classes

    flower_name = cat_to_name[image_file.split('/')[2]]
   # print(flower_name)
   # flower_image = imshow(process_image(image_file))
   # print(flower_image)
    
    probs, classes = predict(image_file, model)
   # plt.figure(figsize=(5,4))
   # sns.barplot(x=probs, y =classes, color=sns.color_palette()[0])
   # plt.show()
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Test an image classifier')
    parser.add_argument('--checkpoint_file_path', type=str, required=True, default='checkpoint.pth', help='File path of model checkpoint')
    parser.add_argument('--image_file_path', type=str, required=True, default='flowers/test/1/image_06760.jpg', help='Number of epochs in training')
    parser.add_argument('--topk', type=int, required=True, default=5, help='Number of most probable classes to predict')
    parser.add_argument('--device', type=str, required=True, default='cuda', help='Set device to either cuda or cpu')

    args = parser.parse_args()                    
    # Functions

    # Set parameters
    print('Default parameters for model are:')
    print('Checkpoint file path: {}'.format(args.checkpoint_file_path))
    print('Image file path: {}'.format(args.image_file_path))
    print('Top k most probable classes: {}'.format(args.topk))
    print('Default device is {}'.format(args.device))

    
    device_to_use(args.device)
    
    # Load label mapping
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    model2 = load_checkpoint(args.checkpoint_file_path)
    print(model2)

    # TODO: Implement the code to predict the class from an image file    
    #image = imshow(process_image(args.image_file_path))

    probs, classes = predict(args.image_file_path, model2)
    print('The top {} classes and associated probabilities are:'.format(args.topk))
    print(probs)
    print(classes)

    predict(args.image_file_path, model2, args.topk)

    print_flower_and_classes(args.image_file_path, model2)


 # python predict.py --checkpoint_file_path 'checkpoint.pth' --image_file_path 'flowers/test/1/image_06760.jpg' --topk 5 --device 'cuda'