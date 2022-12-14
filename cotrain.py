import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms

from algorithms import democratic_co_learning, democratic_co_learning_entropy, no_co_learning
from networks import Model1, Model2, Model3, CNN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

iter = 0

import argparse

parser = argparse.ArgumentParser()
# parser.add_argument("--dataset", default="mnist", type=str, help="Dataset to use")
# parser.add_argument("--network", default="model1", type=str, help="Network to use")
parser.add_argument("--co_learning", default="democratic", type=str, help="Co-learning algorithm to use")
# parser.add_argument("--epochs", default=5, type=int, help="Number of epochs")
# parser.add_argument("--batch_size", default=64, type=int, help="Batch size")
# parser.add_argument("--lr", default=0.01, type=float, help="Learning rate")
# parser.add_argument("--momentum", default=0.5, type=float, help="Momentum")
# parser.add_argument("--seed", default=1, type=int, help="Random seed")
args = parser.parse_args()

data_transform = transforms.Compose([
    # transforms.RandomCrop(32, padding=4),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomVerticalFlip(),
    # transforms.RandomRotation(degrees=15),
    # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the CIFAR-10 dataset
cifar10_train = datasets.CIFAR10('.', train=True, download=True,
                                 transform=transforms.ToTensor())
cifar10_test = datasets.CIFAR10('.', train=False, download=True,
                                transform=transforms.ToTensor())



# Create the models
# model1 = Model1(num_classes=10).to(device)
# model1c = Model1(num_classes=10).to(device)
# model2 = Model2(num_classes=10).to(device)
model3 = Model3(num_classes=10).to(device)
model3c = Model3(num_classes=10).to(device)
model4 = CNN().to(device)
model4c = CNN().to(device)
# model3 = ResNet(in_channels=3, out_channels=10).to(device)
models = [model4, model4c]

# fix the random seed
torch.manual_seed(0)

# Split the training data into labeled and unlabeled datasets
labeled_data, unlabeled_data = torch.utils.data.random_split(cifar10_train, [int(0.6 * len(cifar10_train)), len(cifar10_train) - int(0.6 * len(cifar10_train))])


# Train the models using the democratic co-learning method
if args.co_learning == "democratic":
    print("Training using democratic co-learning")
    democratic_co_learning(models, labeled_data, unlabeled_data, iter)
elif args.co_learning == "democratic_entropy":
   democratic_co_learning_entropy(models, labeled_data, unlabeled_data, iter)
elif args.co_learning == "no_co_learning":
    print("Training without co-learning")
    no_co_learning(models, labeled_data, unlabeled_data, iter)
   


test_loader = torch.utils.data.DataLoader(cifar10_test, batch_size=64, shuffle=False)

  # Compute the accuracy of the models' predictions on the test set
for model in models:
    model.eval()

test_accuracy = 0.0
for data, target in test_loader:
    data = data.to(device)
    target = target.to(device)
    for model in models:
        output = model(data)
        probabilities = model.softmax(output)
        # Use the cross_entropy function to compute the accuracy of the model's predictions
        _, predicted_classes = torch.max(probabilities, dim=1, keepdim=False)

        # Compute the number of correct predictions made by the models on the test set
        test_accuracy += torch.sum(predicted_classes == target)

# Compute the average test accuracy across all models
test_accuracy /= len(models) * len(test_loader.dataset)

# print the accuarcy 
print("Test accuracy: ", test_accuracy.item())


