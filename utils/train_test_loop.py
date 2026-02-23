# from zmq import device
import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset

import pandas as pd
import numpy as np
import random
import time


seed = 69  # You can choose any integer seed
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

'''# If using CUDA, set the seed for GPU as well
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
'''

def train_and_evaluate(
        model, trainloader, testloader, optimizer, loss_fn, num_epochs, 
        pseudolabel_model=None, pseudo_label_loc=3, phi=0.8, sound=10, seed=42):
    
    """Train the model and evaluate on the test set, returning epoch-by-epoch
    results.  
    
    Parameters
    ----------
    model: torch.nn.Module
        The neural network model to be trained.
    trainloader: torch.utils.data.DataLoader
        DataLoader for the training dataset.
    testloader: torch.utils.data.DataLoader
        DataLoader for the test dataset.
    optimizer: torch.optim.Optimizer
        The optimizer to use for training.
    loss_fn: torch.nn.Module
        The loss function to use for training.
    num_epochs: int
        The number of epochs to train the model.
    pseudolabel_model: torch.nn.Module, optional (default None)
        An optional model used for generating pseudo-labels.
    pseudo_label_loc: int, optional (default 3)
        The location/index in the dataset tensors where pseudo-labels are
        stored. This location is set in dataset.py. If changes are made in 
        dataset.py, this parameter should be updated accordingly.
    phi: float, optional (default 0.8)
        Only for PiCO loss. Hyperparameter to average predictions epoch by epoch
    sound: int, optional (default 10)
        The frequency of printing epoch results.
    seed: int, optional (default 42)
        The random seed for reproducibility (default is 42)

    Returns
    -------
    model: torch.nn.Module
        The trained model after all epochs.
    results_df: pandas.DataFrame
        A DataFrame containing epoch-by-epoch results including train loss,
        train accuracy, test accuracy, detached losses, learning rates, and
        epoch times.
    """

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Initialize a list to store epoch data
    results = []

    # Only when debbuging
    # torch.autograd.set_detect_anomaly(True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch+1}/{num_epochs}", flush=True, end='\r')

        start_time = time.time()
        model.train()

        running_loss = 0.0
        correct_train = 0

        for inputs, wl, vl, cl, targets, indices  in trainloader:

            # Move data to the same device as the model
            inputs = inputs.to(device)
            wl, targets = wl.to(device), targets.to(device)
            if pseudolabel_model == 'PiCO':
                vl, cl = vl.to(device), cl.to(device)
                #indices = indices.to(device)

 
            optimizer.zero_grad()
            outputs = model(inputs)

            # Compute loss. PiCO need the pseudolabels too.
            if pseudolabel_model == 'PiCO':
                loss = loss_fn.forward(outputs, wl, cl)
            else:
                loss = loss_fn.forward(outputs, wl) 

            # Model update.
            loss.backward()
            optimizer.step()

            # Update batch's loss and accuracy
            #running_loss += loss.item()
            running_loss += loss.item() * inputs.size(0)   # 按 batch 样本数加权

            # Convert outputs and targets to class indices for acc. calculation
            _, preds = torch.max(outputs, dim=1)
            _, true = torch.max(targets, dim=1)
            correct_train += torch.sum(preds == true)

            # For PiCO loss, update pseudolabels after each epoch
            # PiCO needs to modify the pseudolabels in the dataset
            if pseudolabel_model == 'PiCO':
                with torch.no_grad():
                    # 1. Best class prediction per file
                    max_idx = (outputs * vl).argmax(dim=1)

                    # 2. Pseudolabel update. This is the characteristic step
                    # of PiCO.
                    best_class = torch.zeros_like(outputs)
                    best_class[torch.arange(outputs.size(0)), max_idx] = 1
                    # Warning: Here we assume that the location of pseudolabels
                    # in trainloader.dataset.tensors has been correctly stored
                    # in pseudo_label_loc. If not, the innapropriate tensor
                    # will be modified, which can lead to errors.
                    """ trainloader.dataset.tensors[
                        pseudo_label_loc][indices] = phi * best_class
                    trainloader.dataset.tensors[
                        pseudo_label_loc][indices, preds] += (1 - phi) """
                    pseudo = trainloader.dataset.tensors[pseudo_label_loc]   # 通常在 CPU
                    dev_p = pseudo.device

                    idx = indices.to(dev_p)
                    preds_p = preds.detach().to(dev_p).long()
                    best_class_p = best_class.detach().to(dev_p)

                    pseudo[idx] = phi * best_class_p
                    pseudo[idx, preds_p] += (1 - phi)



        # Compute epoch's loss and accuracy
        train_acc = correct_train.double() / len(trainloader.dataset)
        train_loss = running_loss / len(trainloader.dataset)

        # Evaluate the model on the test set
        model.eval()
        correct_test = 0
        with torch.no_grad():
            for inputs, targets in testloader: 
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)

                _, preds = torch.max(outputs, dim=1)
                _, true = torch.max(targets, dim=1)
                correct_test += torch.sum(preds == true)

        test_acc = correct_test.double() / len(testloader.dataset)

        # Calculate detached loss 
        detached_train_loss = 0.0
        detached_test_loss = 0.0
        with torch.no_grad():
            det_loss_fn = torch.nn.CrossEntropyLoss()  
            for inputs, wl, vl, cl, targets, indices in trainloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                #detached_train_loss += det_loss_fn(outputs, targets).item()
                detached_train_loss += det_loss_fn(outputs, targets).item() * inputs.size(0)

            detached_train_loss /= len(trainloader.dataset)

            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                #detached_test_loss += det_loss_fn(outputs, targets).item()
                detached_test_loss += det_loss_fn(outputs, targets).item() * inputs.size(0)
            detached_test_loss /= len(testloader.dataset)

        # Get the actual learning rate from the optimizer
        actual_lr = optimizer.param_groups[0]['lr']
        
        epoch_time = time.time() - start_time

        # Add summary of epoch results to the list
        epoch_data = {
            'epoch': epoch + 1,
            'pseudolabel_model': pseudolabel_model,
            'train_loss': train_loss,
            'train_acc': train_acc.item(),
            'test_acc': test_acc.item(),
            'train_detached_loss': detached_train_loss,
            'test_detached_loss': detached_test_loss,
            'actual_lr': actual_lr,
            'epoch_time': epoch_time,
        }
        results.append(epoch_data)

        if epoch % sound == sound - 1:
            print(f'Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, '
                  f'Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, '
                  f'Train Detached Loss: {detached_train_loss:.4f}, '
                  f'Test Detached Loss: {detached_test_loss:.4f}, '
                  f'Learning Rate: {actual_lr:.6f}, '
                  f'Epoch Time: {epoch_time:.2f} seconds')

    # Convert results to DataFrame at the end
    results_df = pd.DataFrame(results)

    return model, results_df