import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from keras.datasets import mnist
import matplotlib.pyplot as plt
import seaborn as sns
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Network simulation:
dt = 1e-3
T = 200
ref = 4e-3
batch = 500
usth = 1.1
bmin1 = -25
bmax1 = 25
bmin2 = -25
bmax2 = 25
w_E = 1
rl = 0.001
n_i, n_h1, n_h2, n_o, n_e1, n_e2, n_l = 784, 1000, 1000, 10, 10, 10, 10
tau_syn, tau_s, tau_d = 4e-3, 1e-3, 5e-3
decay_syn, decay_s, decay_d = np.exp(-dt/tau_syn), np.exp(-dt/tau_s), np.exp(-dt/tau_d)

zero = torch.zeros(1, device=device)
one = torch.ones(1, device=device)

# membrane potential update
def LIF(weight, x, rf, I, us):
    I = I * decay_syn + torch.matmul(torch.sign(weight), x)
    us = us * decay_s
    us = torch.where(rf>0., zero, us + I)
    sp = (us >= usth).float()
    us = us * (1. - sp)
    rf = rf + sp * ref - dt
    F.relu(rf, inplace=True)
    
    return rf, I, us, sp

def Boxcar(I, b1, b2):   
    return torch.where((I > b1) | (I < b2), zero, one) 
    
def binary_score(w):
    score = torch.mean((1 - torch.abs(w)) ** 2).item()
    return score

class eWB():    
    def __init__(self):
        super(eWB, self).__init__()
        self.w_hi = torch.empty(n_h1, n_i, device=device)
        self.w_hh = torch.empty(n_h2, n_h1, device=device)
        self.w_oh = torch.empty(n_o, n_h2, device=device)
        self.gp_h1 = torch.empty(n_h1, n_e1, device=device)
        self.gp_h2 = torch.empty(n_h2, n_e1, device=device)
        
        nn.init.xavier_uniform_(self.w_hi)
        nn.init.xavier_uniform_(self.w_hh)
        nn.init.xavier_uniform_(self.w_oh)
        nn.init.xavier_uniform_(self.gp_h1)
        nn.init.xavier_uniform_(self.gp_h2)
        self.gp_h1 = torch.sign(self.gp_h1)
        self.gp_h2 = torch.sign(self.gp_h2)
         
        ### Lamda
        self.lam_hi = torch.zeros(n_h1, n_i, device=device)
        self.lam_hh = torch.zeros(n_h2, n_h1, device=device)
        self.lam_oh = torch.zeros(n_o, n_h2, device=device)
              
    def train(self, x, y):
        self.x = x
        self.y = y
        
        rf_i = sp_i = torch.zeros(n_i, 1, device=device)
        rf_h1= I_h1 = ud_h1 = us_h1 = sp_h1 = torch.zeros(n_h1, 1, device=device)
        rf_h2 = I_h2 = ud_h2 = us_h2 = sp_h2 = torch.zeros(n_h2, 1, device=device)
        rf_o = I_o = ud_o = us_o = sp_o = torch.zeros(n_o, 1, device=device)
        us_e1 = us_e2 = sp_l = sp_e1 = sp_e2 = torch.zeros(n_o, 1, device=device)
        
        for t in range(200):
            sp_i[:, 0] = (self.x > torch.cuda.FloatTensor(n_i).uniform_()).float()
            sp_l[:, 0] = (self.y > torch.cuda.FloatTensor(n_o).uniform_()).float()
            sp_i = torch.where(rf_i>0., zero, sp_i)
            rf_i = rf_i + sp_i * ref - dt
            F.relu(rf_i, inplace=True)
            
            rf_h1, I_h1, us_h1, sp_h1 = LIF(self.w_hi, sp_i, rf_h1, I_h1, us_h1)
            rf_h2, I_h2, us_h2, sp_h2 = LIF(self.w_hh, sp_h1, rf_h2, I_h2, us_h2)
            rf_o, I_o, us_o, sp_o = LIF(self.w_oh, sp_h2, rf_o, I_o, us_o)
            
            ### Error neuron
            us_e1 = us_e1 + w_E * (sp_o - sp_l)
            us_e2 = us_e2 + w_E * (sp_l - sp_o)
            sp_e1 = (us_e1 >= usth).float()
            us_e1 = us_e1 - sp_e1 * usth       
            sp_e2 = (us_e2 >= usth).float()
            us_e2 = us_e2 - sp_e2 * usth  
            
            ### Dendritic potential change by feedback
            ud_h1 = ud_h1 * decay_d + torch.matmul(self.gp_h1, sp_e1 - sp_e2)
            ud_h2 = ud_h2 * decay_d + torch.matmul(self.gp_h2, sp_e1 - sp_e2)
            ud_o = ud_o * decay_d + w_E * (sp_e1 - sp_e2)
            
            ### Weight update
            self.w_hi = self.w_hi - rl * (ud_h1 + 0.0005 * self.lam_hi * -self.w_hi) * Boxcar(I_h1, bmax1, bmin1) * sp_i.transpose(0, 1)
            self.w_hh = self.w_hh - rl * (ud_h2 + 0.0005 * self.lam_hh * -self.w_hh) * Boxcar(I_h2, bmax2, bmin2) * sp_h1.transpose(0, 1)
            self.w_oh = self.w_oh - rl * (ud_o + 0.0005 * self.lam_oh * -self.w_oh) * Boxcar(I_o, bmax1, bmin1) * sp_h2.transpose(0, 1)
                                    
            self.w_hi = torch.clamp(self.w_hi, min=-1., max=1.)
            self.w_hh = torch.clamp(self.w_hh, min=-1., max=1.)
            self.w_oh = torch.clamp(self.w_oh, min=-1., max=1.)                
      
            ### Lamda update
            self.lam_hi = self.lam_hi + 0.2 * rl * torch.abs(1. - self.w_hi ** 2) * Boxcar(I_h1, bmax1, bmin1) * sp_i.transpose(0, 1)
            self.lam_hh = self.lam_hh + 0.2 * rl * torch.abs(1. - self.w_hh ** 2) * Boxcar(I_h2, bmax2, bmin2) * sp_h1.transpose(0, 1)
            self.lam_oh = self.lam_oh + 0.2 * rl * torch.abs(1. - self.w_oh ** 2) * Boxcar(I_o, bmax1, bmin1) * sp_h2.transpose(0, 1)

    def test(self, z):
        self.z = z
        rf_i = sp_i = torch.zeros(batch, n_i, 1, device=device)
        rf_h1= I_h1 = us_h1 = sp_h1 = torch.zeros(batch, n_h1, 1, device=device)
        rf_h2 = I_h2 = us_h2 = sp_h2 = torch.zeros(batch, n_h2, 1, device=device)
        rf_o = I_o = us_o = sp_o = output = torch.zeros(batch, n_o, 1, device=device)

        for t in range(200):          
            sp_i[:, :, 0] = (self.z > torch.cuda.FloatTensor(batch, n_i).uniform_()).float()
            sp_i = torch.where(rf_i>0., zero, sp_i)
            rf_i = rf_i + sp_i * ref - dt
            F.relu(rf_i, inplace=True)
            
            rf_h1, I_h1, us_h1, sp_h1 = LIF(self.w_hi, sp_i, rf_h1, I_h1, us_h1)
            rf_h2, I_h2, us_h2, sp_h2 = LIF(self.w_hh, sp_h1, rf_h2, I_h2, us_h2)
            rf_o, I_o, us_o, sp_o = LIF(self.w_oh, sp_h2, rf_o, I_o, us_o)
            output = output + sp_o
        
        return output

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = torch.from_numpy(x_train.reshape(60000, 784).transpose()).float().cuda() + 10
x_test = torch.from_numpy(x_test.reshape(10000, 784)).float().cuda() + 10

model = eWB()

start = time.time()
print('@@@@@ start @@@@@')
### Learning
n_train, n_test, epoch, acc_step, batch = 60000, 10000, 25, 200, 500
step = []
accuracy = []
bi_score_w_hi = []
bi_score_w_hh = []
bi_score_w_oh = []

for epo in range(epoch):
    for itr in range(n_train):
        x = x_train[:, itr] * dt
        y = torch.cuda.FloatTensor(n_o).fill_(0.)
        y[y_train[itr]] = 1/ref * dt

        model.train(x, y)  

        if np.mod(itr, acc_step) == 0:         
            spike_count = torch.cuda.FloatTensor(n_test, n_o, 1).fill_(0.)

            for itr2 in range(int(n_test/batch)):
                inputs = x_test[itr2 * batch:itr2 * batch + batch, :] * dt
                spike_count[itr2 * batch:itr2 * batch + batch, :, :] = model.test(inputs)
            
            count = np.sum(y_test[0:n_test].reshape(-1,1) - spike_count.cpu().numpy().argmax(axis=1) == 0.)

            print("epoch: %02d"%(epo), end='   ')
            print("step: %05d"%(itr), end='   ')
            print("acc:", "%.4f" %(count/n_test), end='   ')
            print("bs_w_hi:", "%.3f" %(binary_score(model.w_hi)), end='   ')
            print("bs_w_hh:", "%.3f" %(binary_score(model.w_hh)), end='   ')
            print("bs_w_oh:", "%.3f" %(binary_score(model.w_oh)), end='   ')
            print("time: ", "%.1f" %(time.time() - start))
            
            step.append(itr + epo * n_train)
            accuracy.append(count/n_test)
            bi_score_w_hi.append(binary_score(model.w_hi))
            bi_score_w_hh.append(binary_score(model.w_hh))
            bi_score_w_oh.append(binary_score(model.w_oh))
