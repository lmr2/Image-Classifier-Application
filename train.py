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


# Data load, transform, and preparation
def data_transform(data_directory, batch):

    data_dir = data_directory
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # TODO: Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    validation_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=batch, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch, shuffle=False)

    return train_data, train_loader, validation_loader, test_loader

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
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
# Load label mapping
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
# Load model vgg13
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
        

def model_classifier(input_size, hidden_units, output_size):
# Change number of input features and output features to align with data
    classifier = nn.Sequential(OrderedDict([
                              ('fcl', nn.Linear(input_size, hidden_units)), #change features
                              ('relu', nn.ReLU()),
                              ('dropout1', nn.Dropout(0.1)),
                              ('fc2', nn.Linear(hidden_units, output_size)), #change features
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
    return classifier
    # Replace pre-trained classifier
    # model.classifier = classifier

def optimize(learning_rate):
# TODO: Build and train your network
# Set loss function and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    return criterion, optimizer

# TODO: Build and train your network
# Train classifier

def train_classifier(device, epochs, train_loader, steps, running_loss, print_every):
        
    start_time = time.time()
    model.to(device)
    
    for i in range(epochs):
        for inputs, labels in train_loader:
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            #time1 = time.time()
            #print("Time 1 is {}".format(round((time1-start_time)/60,2)))

            if steps % print_every ==0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validation_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        #Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                        #time2 = time.time()
                        #print("Time 2 is {}".format(round((time2-start_time)/60,2)))

                print(f"Epoch {i+1}/{epochs}.. "
                      f"Training loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {test_loss/len(validation_loader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(validation_loader):.3f}")
                running_los = 0
                model.train()

    end_time = time.time()
    print("Total model run time is {}".format(round((end_time-start_time)/60,2)))

    
def testing_classifier(model, test_loader, criterion, batch, device):
    # TODO: Do validation on the test set

    start_time = time.time()

    correct = 0
    total = 0
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print("Accuracy of network on test data is: {}%".format(round(correct/total*100),2))

    end_time = time.time()
    print("Total model run time is {} min".format(round((end_time-start_time)/60,2)))


def save_checkpoint(train_data, model_name, model, epochs, optimizer):
    # TODO: Save the checkpoint 
    model.class_to_idx = train_data.class_to_idx

    checkpoint = {'architecture': model_name,
                  'classifier': model.classifier,
                  'class_to_idx': model.class_to_idx,
                  'state_dict': model.state_dict(),
                  'epochs': epochs,
                  'optimizer': optimizer}

    torch.save(checkpoint, 'checkpoint.pth')
    #'./checkpoint.pth'
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Train an image classifier')
    parser.add_argument('--data_directory', type=str, required=True, help='File path or directory of data')
    parser.add_argument('--batch', type=int, required=True, default=32, help='Batch size')
    
    parser.add_argument('--device', type=str, required=True, default='cuda', help='Set device to either cuda or cpu')

    parser.add_argument('--model_type', type=str, required=True, default='vg13', help='Set model type to vg11, vg13, or vg16 pretrained model')
                       
    parser.add_argument('--input_size', type=int, default=25088, help='Input layer size')
    parser.add_argument('--hidden_units', type=int, required=True, default=500, help='Hidden layer size')
    parser.add_argument('--output_size', type=int, default=102, help='Output layer size')    
    
    parser.add_argument('--learning_rate', type=float, required=True, default=0.001, help='Optimizer learning rate')

    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs in training')
    parser.add_argument('--steps', type=int, default=0, help='Number of steps in training')
    parser.add_argument('--running_loss', type=int, default=0, help='Running loss in training')
    parser.add_argument('--print_every', type=int, default=50, help='Print training results every this number of times')

    args = parser.parse_args()
    # Functions

    # Set parameters
    print('Default parameters for model are:')
    print('Data directory: {}'.format(args.data_directory))
    print('Batch size: {}'.format(args.batch))
    
    print('Default device is {}'.format(args.device))
    
    print('Default model type is {}'.format(args.model_type))

    print('Default input layer size is {}'.format(args.input_size))
    print('Default hidden layer size is {}'.format(args.hidden_units))
    print('Default output layer size is {}'.format(args.output_size))
    
    print('Default learning rate is {}'.format(args.learning_rate))
    
    print('Number of epochs: {}'.format(args.epochs))
    print('Number of steps: {}'.format(args.steps))
    print('Running loss: {}'.format(args.running_loss))
    print('Print every: {}'.format(args.print_every))
    
    train_data, train_loader, validation_loader, test_loader = data_transform(args.data_directory, args.batch)

    device_to_use(args.device)
    # Load label mapping
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        
    # Load model vgg13
    model = model_type(args.model_type)
    
    classifier = model_classifier(args.input_size, args.hidden_units, args.output_size)       
    model.classifier = classifier
    
    criterion, optimizer = optimize(args.learning_rate) 
    
    train_classifier(args.device, args.epochs, train_loader, args.steps, args.running_loss, args.print_every)

    testing_classifier(model, test_loader, criterion, args.batch, args.device)

    save_checkpoint(train_data, args.model_type, model, args.epochs, optimizer)

 # python train.py --data_directory 'flowers/' --batch 32 --device 'cuda' --model_type 'vgg13' --hidden_units 500 --learning_rate .001