# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 14:13:00 2018

@author: Christopher
"""

import torch
from torch import nn
from torch.autograd import Variable
from load_data import load_hyp_spectral, load_hyp_spectral_preprocessed, load_MNIST_raw
from matplotlib import pyplot as plt
import torch.nn.functional as F
import pickle 
from utils2 import heatmap

num_epochs = 20
batch_size = 128
learning_rate = 1e-3
data_size = 220

class CNN_autoencoder(nn.Module): # like https://medium.com/@vaibhaw.vipul/building-autoencoder-in-pytorch-34052d1d280c
    def __init__(self):
        super(CNN_autoencoder,self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=2, stride=2),
            nn.ReLU(True),
            nn.Linear(110,10),
            #nn.ReLU(True)
            )

        self.decoder = nn.Sequential(             
            nn.Linear(10,110),
            nn.ReLU(True),
            nn.ConvTranspose1d(1,1,kernel_size=2,stride=2)
            )

    def forward(self,x):
        x = np.reshape(x, newshape=(1,1,220))
        x = self.encoder(x)
        x = self.decoder(x)
        x = np.squeeze(x)
        return x
    
    def encode(self, x):
        x = np.reshape(x, newshape=(1,1,220))
        x = self.encoder(x)
        x = np.squeeze(x)
        return x

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(data_size, 10),
            #nn.ReLU(True)
            #nn.Linear(128, 64),
            #nn.ReLU(True),
            #nn.Linear(64, 32), 
            #nn.ReLU(True), 
            #nn.Linear(32, 10)
            )
        self.decoder = nn.Sequential(
            nn.Linear(10, data_size),
            #nn.ReLU(True),
            #nn.Linear(32, 64),
            #nn.ReLU(True),
            #nn.Linear(64, 128),
            #nn.ReLU(True), 
            #nn.Linear(128, data_size),
            #nn.Tanh()
            )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)
    
    
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(220, 110)
        self.fc21 = nn.Linear(110, 10)
        self.fc22 = nn.Linear(110, 10)
        self.fc3 = nn.Linear(10, 110)
        self.fc4 = nn.Linear(110, 220)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return F.relu(self.fc4(h3))
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar
    
    
    
from scipy.io import loadmat
import numpy as np
f = loadmat("data/indian_pines.mat")
raw_data = np.asarray(f['indian_pines'])
raw_data = np.reshape(raw_data, newshape=(145*145,220))
print(raw_data.shape)


def train():
    data = load_hyp_spectral_preprocessed()
    print("n:", sum([len(data[k]) for k in data]))
    #data = load_MNIST_raw()
    model = autoencoder()
    criterion = nn.MSELoss()
    """
    SGD mostly finds all the same features for all classes as optimal solution, but sometimes not. In these cases, the feature map looks promissing.
    Adam finds solutions with a smaller loss, and the features look also more promissing.
    
    
    """
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print("start")
    losses = []
    for epoch in range(num_epochs):
        epoch_loss = []
        for key in data:
            batch = data[key]
            for sample in batch:
                img = Variable(torch.from_numpy(sample.astype(float)).float())
                # ===================forward=====================
                output = model(img)
                loss = criterion(output, img)
                # ===================backward====================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.data)
        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'
              .format(epoch + 1, num_epochs, loss.data))
        losses.append(np.average(epoch_loss))
    plt.plot(losses)
    #torch.save(model.state_dict(), './sim_autoencoder.pth')
    
    return model


def train_MNIST():
    data = load_hyp_spectral_preprocessed()
    #data, _ = load_MNIST_raw()
    batch_size = 3 # 3 for mnist
    data_batches = np.split(np.asarray(data), batch_size)
    model = autoencoder()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    print("start")
    for epoch in range(num_epochs):
        for b in data_batches:
            batch = b
            for sample in batch:
                img = Variable(torch.from_numpy(sample.astype(float)).float())
                # ===================forward=====================
                output = model(img)
                loss = criterion(output, img)
                # ===================backward====================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'
              .format(epoch + 1, num_epochs, loss.data))
        
    #torch.save(model.state_dict(), './sim_autoencoder.pth')
    return model

reconstruction_function = nn.MSELoss()


def loss_function(recon_x, x, mu, logvar):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    BCE = reconstruction_function(recon_x, x)  # mse loss
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    # KL divergence
    return BCE + KLD


def train_VAE():
    model = VAE()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        for data in raw_data:
            data = np.asarray(data) # normalize the input, bc it would explode otherwise
            data = (data - np.average(data)) / np.std(data)
            img = Variable(torch.from_numpy(data.astype(float)).float())
            # ===================forward=====================
            recon, mu, logvar = model(img)
            loss = loss_function(recon, img, mu, logvar)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'
              .format(epoch + 1, num_epochs, loss.data))
        
    #torch.save(model.state_dict(), './sim_autoencoder.pth')
    return model


def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def autoencoder_features(model):
    data = load_hyp_spectral()
    avgs = {}
    for c in range(1,17):
        c = str(c)
        results = []
        for row in data[c]:
            row = Variable(torch.from_numpy(row.astype(float)).float())
            result = model.encode(row).data.numpy()
            #print(result.shape)
            results.append(result)
        avg = np.average(results, axis=0)
        #print(c, avg[0:5])
        avgs[c] = avg
    #print(avgs)
    #print([list(avgs[key]) for key in avgs])
    heatmap([list(avgs[key]) for key in avgs])
    save_object(avgs, "obj/autoencoder_features_10_e50.pkl")
  
def autoencoder_features2(model):
    data = load_hyp_spectral_preprocessed()
    avgs = {}
    print(data.keys())
    for c in range(1,17):
        #c = str(c)
        avg = np.zeros(220,)
        if c in data.keys():
            for row in data[c]:
                avg += np.asarray(row)
            avg /= len(data[c])
            print(avg[0:5])
            avg = Variable(torch.from_numpy(avg.astype(float)).float())
            result = model.encode(avg).data.numpy()
            
            #print(c, avg[0:5])
            avgs[str(c)] = result
    #print(avgs)
    #print([list(avgs[key]) for key in avgs])
    print(avgs)
    heatmap([list(avgs[key]) for key in avgs])
    save_object(avgs, "obj/autoencoder_features_10_e50.pkl")

def VAE_features(model):
    data = load_hyp_spectral()
    avgs = {}
    for c in range(1,17):
        c = str(c)
        results = []
        for row in data[c]:
            row = Variable(torch.from_numpy(row.astype(float)).float())
            result, _ = model.encode(row)
            results.append(result.data.numpy())
        avg = np.average(results, axis=0)
        print(c, avg[0:5])
        avgs[c] = avg
    #print(avgs)
    print([list(avgs[key]) for key in avgs])
    heatmap([list(avgs[key]) for key in avgs])
    save_object(avgs, "obj/VAE_features_10_e50.pkl")
     

def autoencoder_features_mnist(model):
    data, labels = load_MNIST_raw()
    avgs = {}
    for c in range(10):
        results = []
        for index, l in enumerate(labels):
            if l == c:
                
                
                row = Variable(torch.from_numpy(data[index].astype(float)).float())
                result = model.encode(row).data.numpy()
                results.append(result)
        avg = np.average(results, axis=0)
        print(c, avg[0:5])
        avgs[c] = avg
    #print(avgs)
    save_object(avgs, "obj/autoencoder_features_mnist_10_e50.pkl")
    
    
def test(model):
    data = load_hyp_spectral_preprocessed()
    for c in data:
        
        #print(data[c][0])
        plt.figure(c)
        plt.subplot(2,1,1)
        heatmap(list([data[c][0][0:10]]), title="original", x_size=10, y_size = 0.2)
        
        row = Variable(torch.from_numpy(data[c][0].astype(float)).float())
        #print(model.encode(row))
        #print(model(row))
        #print("data", np.asarray(model(row).data[0:10]))
        plt.subplot(2,1,2)
        heatmap([np.asarray(model(row).data[0:10])],title = "reconstruction", x_size=10, y_size = 0.2)
        
    
m = train()
#test(m)

autoencoder_features2(m)
#VAE_features(m)
#autoencoder_features_mnist(m)










