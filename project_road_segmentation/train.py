# Authors: Gianni Lodetti, Luca Bracone, Omid Karimi
# Function to train models

import torch
from models import *
from torch import nn
from torch import optim
from losses import *
from sklearn.metrics import f1_score

def train_model(model, train_input, train_target, LEARNING_RATE, mini_batch_size, nbr_epochs, DEVICE,
                opt="Adam", loss_func="CrossEntropy"):
    """Trains the model

    Args:
        model (torch.nn.Module): A torch neural network
        train_input (4d tensor): 
            A torch tensor of shape (num_images, num_channels, width, height)
        train_target (4d tensor): 
            A torch tensor of shape (num_images, 1, width, height) that
            represents the desired output
        LEARNING_RATE (float): 
            The coefficient by which we update the weights at each training step
        mini_batch_size (int): 
            The mini batch size
        nbr_epochs (int): 
            How many iterations to train for
        DEVICE (str): cuda or cpu
        opt (str, optional): 
            The optimizer of choice. Defaults to "Adam".
        loss_func (str, optional): 
            The loss function of choice. Defaults to "CrossEntropy".
    """
    # Select loss function
    if loss_func == "CrossEntropy":
        criterion = nn.BCEWithLogitsLoss()
        # criterion = nn.CrossEntropyLoss()
    elif loss_func == "Jaccard":
        criterion = Jaccard()

    # Select optimizer
    if opt == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)  
    elif opt == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Number of epochs: {nbr_epochs}")
    for e in range(nbr_epochs):
        # We do this with mini-batches
        print(f"Current epoch: %d" % e)
        for b in range(0, train_input.size(0), mini_batch_size):  # mini_batch_size = num of patches at each iteration
            if b + mini_batch_size > train_input.size(0):
                batch_input = train_input.narrow(dim=0, start=b, length=train_input.size(0) - b).to(DEVICE)
                batch_label = train_target.narrow(dim=0, start=b, length=train_input.size(0) - b).to(DEVICE)
            else:
                batch_input = train_input.narrow(dim=0, start=b, length=mini_batch_size).to(DEVICE)
                batch_label = train_target.narrow(dim=0, start=b, length=mini_batch_size).to(DEVICE)

            preds = model(batch_input)

            loss = criterion(preds, batch_label.float().unsqueeze(1))

            if (b % 40 == 0):
                print(f"Iteration: %d" % b)
                print(f"loss: {loss}")
                print("=============================================")
            model.zero_grad()
            loss.backward()  
            optimizer.step()
