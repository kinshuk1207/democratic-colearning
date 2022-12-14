import numpy as np
import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class Model1(nn.Module):
  def __init__(self, num_classes):
    super(Model1, self).__init__()
    self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
    self.relu1 = nn.ReLU()
    self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
    self.relu2 = nn.ReLU()
    self.pool = nn.MaxPool2d(2, 2)
    self.fc1 = nn.Linear(64 * 16 * 16, 1024)
    self.relu3 = nn.ReLU()
    self.fc2 = nn.Linear(1024, num_classes)
    self.softmax = nn.Softmax(dim=1)
    
    self.loss = nn.CrossEntropyLoss()
    self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)
  
  def entropy(self, output):
    # Compute the entropy of the model's predictions
    entropy = -torch.sum(output * torch.log(output))
    return entropy
  
  def forward(self, x):
    x = x.to(device)
    out = self.conv1(x)
    out = self.relu1(out)
    out = self.conv2(out)
    out = self.relu2(out)
    out = self.pool(out)
    out = out.view(out.size(0), -1)
    out = self.fc1(out)
    out = self.relu3(out)
    out = self.fc2(out)
    out = self.softmax(out)
    return out

import torch.nn.functional as F    

class Model2(nn.Module):
  def __init__(self, num_classes):
    super(Model2, self).__init__()
    self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
    self.relu1 = nn.ReLU()
    self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
    self.relu2 = nn.ReLU()
    self.pool = nn.MaxPool2d(2, 2)
    self.fc1 = nn.Linear(32 * 16 * 16, 512)
    self.relu3 = nn.ReLU()
    self.fc2 = nn.Linear(512, num_classes)
    self.softmax = nn.Softmax(dim=1)
 
    self.loss = nn.CrossEntropyLoss()
    self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)

  def entropy(self, output):
    # Compute the entropy of the model's predictions
    entropy = -torch.sum(output * torch.log(output))
    return entropy 
  
  def forward(self, x):
    x = x.to(device)
    out = self.conv1(x)
    out = self.relu1(out)
    out = self.conv2(out)
    out = self.relu2(out)
    out = self.pool(out)
    out = out.view(out.size(0), -1)
    out = self.fc1(out)
    out = self.relu3(out)
    out = self.fc2(out)
    out = self.softmax(out)
    return out

class Model3(nn.Module):
  def __init__(self, num_classes):
    super(Model3, self).__init__()
    self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
    self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
    self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
    self.relu1 = nn.ReLU()
    self.relu2 = nn.ReLU()
    self.relu3 = nn.ReLU()
    self.relu4 = nn.ReLU()
    self.pool = nn.MaxPool2d(2, 2)
    self.fc1 = nn.Linear(256 * 16 * 16, 1024)
    self.fc2 = nn.Linear(1024, num_classes)
    self.softmax = nn.Softmax(dim=1)
    
    self.loss = nn.CrossEntropyLoss()
    self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)
    # self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0003)
  
  def entropy(self, output):
    # Compute the entropy of the model's predictions
    entropy = -torch.sum(output * torch.log(output))
    return entropy
  
  def forward(self, x):
    x = x.to(device)
    out = self.conv1(x)
    out = self.relu1(out)
    out = self.conv2(out)
    out = self.relu2(out)
    out = self.conv3(out)
    out = self.relu3(out)
    out = self.conv4(out)
    out = self.relu4(out)
    out = self.pool(out)
    out = out.view(out.size(0), -1)
    out = self.fc1(out)
    out = self.fc2(out)
    out = self.softmax(out)
    return out

class CNN(nn.Module):
   

    def __init__(self):
        
        super(CNN, self).__init__()

        self.conv_layer = nn.Sequential(

            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )


        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 10)
        )
        
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        # elf.optimizer = torch.optim.Adam(self.parameters(), lr=0.0003)
        
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        """Perform forward."""
        
        x = x.to(device)
        # conv layers
        x = self.conv_layer(x)
        
        # flatten
        x = x.view(x.size(0), -1)
        
        # fc layer
        x = self.fc_layer(x)

        return x
    
    
# class ResNet(nn.Module):
#   def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
#     super(ResNet, self).__init__()
#     self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
#     self.bn1 = nn.BatchNorm2d(num_features=out_channels)
#     self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
#     self.bn2 = nn.BatchNorm2d(num_features=out_channels)
#     self.fc = nn.Linear(in_features=out_channels, out_features=10)
#     self.loss = nn.CrossEntropyLoss()
#     self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
  
#   def forward(self, x):
#     x = x.to(device)
#     x_d = self.conv1(x)
#     out = self.conv1(x)
#     out = self.bn1(out)
#     out = F.relu(out) + x_d
#     out = self.conv2(out)
#     out = self.bn2(out)
#     # out = out.view(out.size(0), -1)
#     out = self.fc(out)
#     print(out.shape)
#     quit()
#     return out
