#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import torch

from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data

import time

# GPU/CPU setting
def try_gpu(e):
    if torch.cuda.is_available():
        return e.cuda()
    
    return e

# Encoder
# Input dimension is (number of data, symbol length, feature dimension)
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.rnn_1 = nn.LSTM(input_size = 463, hidden_size = 128, batch_first = True, dropout = 0.001, bidirectional = True)
        self.rnn_2 = nn.LSTM(input_size = 256, hidden_size = 32, batch_first = True, dropout = 0.001, bidirectional = True)

        self.mu_dense = nn.Linear(64 * 30, 256)
        self.logvar_dense = nn.Linear(64 * 30, 256)
        
    def forward(self, x):
        x = x.float()
        x, _ = self.rnn_1(x)
        x, _ = self.rnn_2(x)
        x = torch.flatten(x, start_dim = 1)
        
        mu = self.mu_dense(x)
        logvar = self.logvar_dense(x)
        
        return mu, logvar
    
# Decoder
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn1 = nn.LSTM(input_size = 256, hidden_size = 128, batch_first = True, dropout = 0.001)
        self.rnn2 = nn.LSTM(input_size = 128, hidden_size = 128, batch_first = True, dropout = 0.001)
        self.out_dense = nn.Linear(128, 463)
        
    # Decoder layers
    def forward(self, x):
        x, _ = self.rnn1(x)
        x, _ = self.rnn2(x)
        
        x = torch.softmax(self.out_dense(x), dim = 2)
        
        return x
    
# Convert character string to decoder input data
class Decoder_input(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder_input = nn.Linear(463, 256)
        
    def forward(self, x):
        # Input decoder training data
        x = x.float()
        
        # Convert inputtable data type to RNN (number of data, symbol length, feature dimension)
        x = self.decoder_input(x)
        
        return x
        
# VAE class
class VAE(nn.Module):
    def __init__(self, encoder, decoder, dim_converter):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.dim_converter = dim_converter
        
    # Sampling on the latent space
    def sampling(self, mu, logvar):
        std = torch.exp(logvar / 2)
        eps = torch.randn_like(std)
        x_sample = eps.mul(std).add_(mu)

        return x_sample
        
    def forward(self, x, x_2):
        
        # Encoder
        mu, logvar = self.encoder(x)
        x_sample = self.sampling(mu, logvar)
        
        # Convert inputtable data type to RNN
        x = torch.reshape(x_sample, (-1, 1, 256))
        
        # Dimension conversion
        x_2 = self.dim_converter(x_2)
            
        # Combine with latent space vector
        x_decoder_input = torch.cat([x, x_2], dim = 1)
        
        # Decoder
        x = self.decoder(x_decoder_input).float()

        return x, mu, logvar

# Loss function
def loss_estimate(x_true, x_pred, mu, logvar, class_weight):
    x_true = try_gpu(x_true.float())
    x_pred = try_gpu(x_pred.float())

    # Weight according to the ratio of 0 to 1
    loss_weight = torch.zeros(x_true.size(0), x_true.size(1), x_true.size(2))
    loss_weight[x_true == 0] = class_weight[0]
    loss_weight[x_true == 1] = class_weight[1]
    loss_weight = try_gpu(loss_weight.float())

    # Cross entropy
    entropy = nn.BCELoss(weight = loss_weight, reduction = "sum")
    entropy_loss = entropy(x_pred, x_true)

    # KL divergence
    kl_loss =  torch.sum(- logvar + torch.exp(logvar) + mu ** 2 - 1)
    kl_loss *= 0.5

    # Total loss
    vae_loss = entropy_loss + kl_loss

    # Accuracy
    total_elements = x_true.size(0) * x_true.size(1)
    same_elements = torch.sum(torch.argmax(x_true, 2) == torch.argmax(x_pred, 2)).item()

    return vae_loss, same_elements, total_elements

if __name__ == "__main__":
    
    st = time.time()

    model_path = "../result/fragment_VAE.pth"
    history_path = "../result/history_fragment_VAE.csv"
    
    
    X_train = np.load("../data/train_fragment_selfies_onehot.npy")
    X_validation = np.load("../data/validation_fragment_selfies_onehot.npy")

    # Data weight
    weight = torch.from_numpy(np.array([0.501082, 231.5])).float()

    # Data loader
    trainloader = torch.utils.data.DataLoader(X_train, batch_size = 2048, shuffle=True, num_workers = 4, pin_memory = True)
    validationloader = torch.utils.data.DataLoader(X_validation, batch_size = 2048, shuffle=False, num_workers = 4, pin_memory = True)

    model = VAE(Encoder(), Decoder(), Decoder_input())
    model = try_gpu(model)

    # Optimization function
    optimizer = optim.Adam(model.parameters(), lr = 0.0001)

    es_num = 30
    es_count = 1
    epochs = 3000

    max_acc = None

    output_history = pd.DataFrame(np.zeros((epochs, 4)), columns = ["loss", "acc", "val_loss", "val_acc"])

    for epoch in list(range(epochs)):
        print(epoch)    
        model.train()

        # Loss fuction values
        running_loss = 0.0
        running_same = 0.0
        running_total = 0.0

        for i, data in enumerate(trainloader):
            data = try_gpu(data.float())

            # Optimizer initialization
            optimizer.zero_grad()

            x_pred, mu, logvar = model(data, data[:, :-1, :])

            vae_loss, same_elements, total_elements = loss_estimate(data, x_pred, mu, logvar, weight)

            vae_loss.backward()

            running_loss += np.float(vae_loss.data)
            running_same += same_elements
            running_total += total_elements

            optimizer.step()

        # Save the history
        output_history.iloc[epoch, 0] = running_loss 

        accuracy = running_same / running_total
        output_history.iloc[epoch, 1] = accuracy

        # Evaluation by the validation data
        model.eval()

        # Loss fuction values
        val_running_loss = 0.0
        val_running_same = 0.0
        val_running_total = 0.0

        with torch.no_grad():
            for i, data in enumerate(validationloader):
                data = try_gpu(data.float())
                X_val_pred, val_mu, val_logvar = model(data, data[:, :-1, :])

                val_vae_loss, val_same_elements, val_total_elements = loss_estimate(data, X_val_pred, val_mu, val_logvar, weight)

                val_running_loss += np.float(val_vae_loss.data)

                val_running_same += val_same_elements
                val_running_total += val_total_elements

            # Save the history
            output_history.iloc[epoch, 2] = val_running_loss 

            val_accuracy = val_running_same / val_running_total
            output_history.iloc[epoch, 3] = val_accuracy

        # Output scores by evaluation function and loss function
        output_history.to_csv(history_path)

        # Early stopping

        # Execute only first epoch
        if max_acc == None:
            max_acc = val_accuracy
            es_count += 1
            continue

        # Case for the largest validation accuracy
        if max_acc < val_accuracy:
            max_acc = val_accuracy
            es_count = 1
            torch.save(model.state_dict(), model_path)

            continue

        # Case NOT for the largest validation accuracy
        else:
            # Finish the calculation if the specified number is reached
            if es_num <= es_count:
                break

            else:
                es_count += 1

        if epoch == epochs:
            torch.save(model.state_dict(), model_path)
            output_history.to_csv(history_path)

    et = time.time()
    total_time = et - st
    with open("../result/calculation_time_fragment_VAE.txt", "a") as writer:
        writer.write(str(total_time))
