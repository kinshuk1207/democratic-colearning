import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def democratic_co_learning(models, labeled_data, unlabeled_data, iter):

  # Load labeled data into a dataloader
  labeled_dataloader = torch.utils.data.DataLoader(labeled_data, batch_size=64, shuffle=True)

  # Train both models on the labeled data
  for model in models:
    model.train()
    for (x, y) in labeled_dataloader:
      model.zero_grad()
      x = x.to(device)
      outputs = model(x)
      y = y.to(device)
      loss = model.loss(outputs, y)
      loss.backward()
      model.optimizer.step()

  # Make a dataloader for the unlabeled data
  unlabeled_dataloader = torch.utils.data.DataLoader(unlabeled_data, batch_size=64, shuffle=False)

  # Use the models to make predictions on the unlabeled data
  relabelled_data = []
  predictions = []
  for data, target in unlabeled_dataloader:
    # Initialize a list to store the confidence intervals of the models
    confidence_intervals = []
    

    # Compute the confidence interval of each model's predictions
    for model in models:
      output = model(data)
      probabilities = model.softmax(output)
      confidence_interval = probabilities.max() - probabilities.min()
    #   print(confidence_interval)
      confidence_intervals.append(confidence_interval)

    # Check if a majority of the models agree on the label of the example
    if sum(confidence_intervals) > len(models) / 2:
      # Add the example to the labeled dataset and retrain the models
      relabelled_data.append((data.numpy()))
      
      predictions.append(probabilities.max(1, keepdim=True)[1].squeeze().cpu().numpy())
      
  
  relabeled = torch.utils.data.TensorDataset(torch.from_numpy(relabelled_data[0]), torch.from_numpy(predictions[0]))
  for e in range(1,len(relabelled_data)):
    relabeled = torch.utils.data.ConcatDataset([relabeled, torch.utils.data.TensorDataset(torch.from_numpy(relabelled_data[e]), torch.from_numpy(predictions[e]))])
  
  relabled_dataloader = torch.utils.data.DataLoader(relabeled, batch_size=64, shuffle=True)
  
  for model in models:
    model.train()
    for (x, y) in relabled_dataloader:
      model.zero_grad()
      x = x.to(device)
      outputs = model(x)
      y = y.to(device)
      loss = model.loss(outputs, y)
      loss.backward()
      model.optimizer.step()
      
    
  # Repeat the process until no more examples can be added to the labeled data
  if len(unlabeled_data) > 0 and iter < 20:
      iter += 1
      print("Iteration: ", iter)
      democratic_co_learning(models, labeled_data, unlabeled_data, iter)


def democratic_co_learning_entropy(models, labeled_data, unlabeled_data, iter):

  # Load labeled data into a dataloader

  labeled_dataloader = torch.utils.data.DataLoader(labeled_data, batch_size=64, shuffle=True)

  # Train both models on the labeled data
  for model in models:
    model.train()
    for (x, y) in labeled_dataloader:
      model.zero_grad()
      x = x.to(device)
      outputs = model(x)
      y = y.to(device)
      loss = model.loss(outputs, y)
      loss.backward()
      model.optimizer.step()

  # # Make a dataloader for the unlabeled data
  # unlabeled_dataloader = torch.utils.data.DataLoader(unlabeled_data, batch_size=64, shuffle=False)

  # # Use the models to make predictions on the unlabeled data
  # for data, target in unlabeled_dataloader:
  #   # Initialize a list to store the confidence intervals of the models
  #   confidence_intervals = []

  #   # Compute the confidence interval of each model's predictions
  #   for model in models:
  #     output = model(data)
  #     probabilities = model.softmax(output)
  #     # Use entropy as the measure of confidence
  #     entropy = -torch.sum(torch.mul(probabilities, torch.log2(probabilities)), dim=1)
  #     confidence_interval = entropy.mean()
  #     confidence_intervals.append(confidence_interval)

  #   # Check if a majority of the models agree on the label of the example
  #   # Check if a majority of the models have low entropy value for the example
  #   if sum([1 for ci in confidence_intervals if ci < 0.5]) > len(models) / 2:
  #     # Add the example to the labeled dataset and retrain the models
  #     labeled_data.append((data, target))

  # Repeat the process until no more examples can be added to the labeled data
  if len(unlabeled_data) > 0 and iter < 10:
      iter += 1
      print("Iteration: ", iter)
      democratic_co_learning_entropy(models, labeled_data, unlabeled_data, iter)


def democratic_co_learning_entropy2(models, labeled_data, unlabeled_data, iter):

  # Load labeled data into a dataloader

  labeled_dataloader = torch.utils.data.DataLoader(labeled_data, batch_size=64, shuffle=True)

  # Train both models on the labeled data
  for model in models:
    model.train()
    for (x, y) in labeled_dataloader:
      model.zero_grad()
      x = x.to(device)
      outputs = model(x)
      y = y.to(device)
      loss = model.loss(outputs, y)
      loss.backward()
      model.optimizer.step()

  # Make a dataloader for the unlabeled data
  unlabeled_dataloader = torch.utils.data.DataLoader(unlabeled_data, batch_size=64, shuffle=False)

  # Use the models to make predictions on the unlabeled data
  for data, target in unlabeled_dataloader:
    # Initialize a list to store the confidence intervals of the models
    confidence_intervals = []
    entropies = []

    # Compute the confidence interval and entropy of each model's predictions
    for model in models:
      output = model(data)
      probabilities = model.softmax(output)
      confidence_interval = probabilities.mean()
      confidence_intervals.append(confidence_interval)
      entropy = -(probabilities * torch.log(probabilities)).sum()
      entropies.append(entropy)

    # Check if a majority of the models agree on the label of the example
    # Check if a majority of the models have low entropy value for the example
    if sum([1 for ci in confidence_intervals if ci < 0.5]) > len(models) / 2:
      # Add the example to the labeled dataset and retrain the models
      labeled_data.append((data, target))

  # Repeat the process until no more examples can be added to the labeled data
  if len(unlabeled_data) > 0 and iter < 20:
      iter += 1
      print("Iteration: ", iter)
      democratic_co_learning_entropy2(models, labeled_data, unlabeled_data, iter)
      
def no_co_learning(models, labeled_data, unlabeled_data, iter):

  # Load labeled data into a dataloader

  labeled_dataloader = torch.utils.data.DataLoader(labeled_data, batch_size=64, shuffle=True)

  # Train both models on the labeled data
  for model in models:
    model.train()
    for (x, y) in labeled_dataloader:
      model.zero_grad()
      x = x.to(device)
      outputs = model(x)
      y = y.to(device)
      loss = model.loss(outputs, y)
      loss.backward()
      model.optimizer.step()

  if len(unlabeled_data) > 0 and iter < 20:
      iter += 1
      print("Iteration: ", iter)
      no_co_learning(models, labeled_data, unlabeled_data, iter)
