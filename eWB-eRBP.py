import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from keras.datasets import mnist
import matplotlib.pyplot as plt
import seaborn as sns
import time

torch.cuda.set_device(1)

class eRBP(object):
    
    def __init__(self, batch, n_i, n_h1, n_h2, n_o, n_e1, n_e2, n_l, dt, tau_syn, tau_s, tau_d, ref1, ref2, 
                 usth, bmin1, bmax1, bmin2, bmax2, w_E, p, rl):
        
        ### Architecture
        self.batch = batch
        self.n_i = n_i
        self.n_h1 = n_h1
        self.n_h2 = n_h2
        self.n_o = n_o
        self.n_e1 = n_e1
        self.n_e2 = n_e2
        self.n_l = n_l     

        ### LIF parameters ###
        self.dt = dt
        self.tau_syn = tau_syn
        self.tau_s = tau_s
        self.tau_d = tau_d

        self.ref1 = ref1
        self.ref2 = ref2
        
        self.usth = usth
        self.bmin1 = bmin1
        self.bmax1 = bmax1
        self.bmin2 = bmin2
        self.bmax2 = bmax2
        
        self.p = p
        
        self.rl = rl
        
        ### Neuron
        self.ud_h1 = torch.zeros(self.n_h1, 1).cuda()
        self.us_h1 = torch.zeros(self.n_h1, 1).cuda()        
        
        self.ud_h2 = torch.zeros(self.n_h2, 1).cuda()
        self.us_h2 = torch.zeros(self.n_h2, 1).cuda()

        self.ud_o = torch.zeros(self.n_o, 1).cuda()
        self.us_o = torch.zeros(self.n_o, 1).cuda()
        
        self.us_e1 = torch.zeros(self.n_e1, 1).cuda()
        self.us_e2 = torch.zeros(self.n_e2, 1).cuda()

        self.tus_h1 = torch.zeros(self.batch, self.n_h1, 1).cuda()
        self.tus_h2 = torch.zeros(self.batch, self.n_h2, 1).cuda()
        self.tus_o = torch.zeros(self.batch, self.n_o, 1).cuda()

        ### Weights
        self.w_hi = torch.empty(self.n_h1, self.n_i).cuda()
        self.w_hh = torch.empty(self.n_h2, self.n_h1).cuda()
        self.w_oh = torch.empty(self.n_o, self.n_h2).cuda()
        self.gp_h1 = torch.empty(self.n_h1, self.n_e1).cuda()
        self.gp_h2 = torch.empty(self.n_h2, self.n_e1).cuda()
        
        nn.init.xavier_uniform_(self.w_hi)
        nn.init.xavier_uniform_(self.w_hh)
        nn.init.xavier_uniform_(self.w_oh)
      
        nn.init.xavier_uniform_(self.gp_h1)
        nn.init.xavier_uniform_(self.gp_h2)
                
        self.gp_h1 = torch.sign(self.gp_h1)
        self.gp_h2 = torch.sign(self.gp_h2)
        
        self.w_E = w_E      

        self.wb_hi = torch.empty(self.n_h1, self.n_i).cuda()
        self.wb_hh = torch.empty(self.n_h2, self.n_h1).cuda()
        self.wb_oh = torch.empty(self.n_o, self.n_h2).cuda()
        
        ### Lamda
        self.lam_hi = torch.zeros(self.n_h1, self.n_i).cuda()
        self.lam_hh = torch.zeros(self.n_h2, self.n_h1).cuda()
        self.lam_oh = torch.zeros(self.n_o, self.n_h2).cuda()
        
        ### Current
        self.I_h1 = torch.zeros(self.n_h1, 1).cuda()
        self.I_h2 = torch.zeros(self.n_h2, 1).cuda()
        self.I_o = torch.zeros(self.n_o, 1).cuda()

        self.tI_h1 = torch.zeros(self.batch, self.n_h1, 1).cuda()
        self.tI_h2 = torch.zeros(self.batch, self.n_h2, 1).cuda()
        self.tI_o = torch.zeros(self.batch, self.n_o, 1).cuda()
                
        ### spikes ###       
        self.sp_i = torch.zeros(self.n_i, 1).cuda()
        self.sp_h1 = torch.zeros(self.n_h1, 1).cuda()
        self.sp_h2 = torch.zeros(self.n_h2, 1).cuda()
        self.sp_o = torch.zeros(self.n_o, 1).cuda()
        self.sp_l = torch.zeros(self.n_l, 1).cuda()
        self.sp_e1 = torch.zeros(self.n_e1, 1).cuda()
        self.sp_e2 = torch.zeros(self.n_e2, 1).cuda()

        self.tsp_i = torch.zeros(self.batch, self.n_i, 1).cuda()
        self.tsp_h1 = torch.zeros(self.batch, self.n_h1, 1).cuda()
        self.tsp_h2 = torch.zeros(self.batch, self.n_h2, 1).cuda()
        self.tsp_o = torch.zeros(self.batch, self.n_o, 1).cuda()
        
        ### jump ###
        self.jump = 0
        
        ### decay ###  
        self.decay_syn = np.exp(-self.dt/self.tau_syn)
        self.decay_s = np.exp(-self.dt/self.tau_s)
        self.decay_d = np.exp(-self.dt/self.tau_d)
        
        ### Refractory
        self.rf_i = torch.zeros(self.n_i, 1).cuda()
        self.rf_h1 = torch.zeros(self.n_h1, 1).cuda()
        self.rf_h2 = torch.zeros(self.n_h2, 1).cuda()
        self.rf_o = torch.zeros(self.n_o, 1).cuda()
        
        self.trf_i = torch.zeros(self.batch, self.n_i, 1).cuda()
        self.trf_h1 = torch.zeros(self.batch, self.n_h1, 1).cuda()
        self.trf_h2 = torch.zeros(self.batch, self.n_h2, 1).cuda()
        self.trf_o = torch.zeros(self.batch, self.n_o, 1).cuda()

        ### Cuda variable
        self.zero = torch.zeros(1).cuda()
        self.one = torch.ones(1).cuda()
        
        ### Spikes count
        self.count_i = torch.zeros(self.n_i, 1).cuda()
        self.count_h1 = torch.zeros(self.n_h1, 1).cuda()
        self.count_h2 = torch.zeros(self.n_h2, 1).cuda()
        self.count_o = torch.zeros(self.n_o, 1).cuda()
        self.count_e1 = torch.zeros(self.n_e1, 1).cuda()     
        self.count_e2 = torch.zeros(self.n_e2, 1).cuda()     
         
    def boxcar(self, I, b1, b2):   
        return torch.where((I > b1) | (I < b2), self.zero, self.one) 
        
    def learning_step(self):
        ### input jump ###
        if self.jump == 1:
            self.I_h1 *= 0.
            self.ud_h1 *= 0.
            self.us_h1 *= 0.   

            self.I_h2 *= 0.
            self.ud_h2 *= 0.
            self.us_h2 *= 0.
            
            self.I_o *= 0.
            self.ud_o *= 0.
            self.us_o *= 0.
            self.us_e1 *= 0.
            self.us_e2 *= 0.
            
            self.rf_i *= 0.
            self.rf_h1 *= 0.
            self.rf_h2 *= 0.
            self.rf_o *= 0.
        
            self.jump = 0
        
        self.wb_hi = torch.sign(self.w_hi)
        self.wb_hh = torch.sign(self.w_hh)
        self.wb_oh = torch.sign(self.w_oh)
        
        ## Neuron firing:
        ### Input neuron
        self.sp_i = torch.where(self.rf_i>0., self.zero, self.sp_i)
        self.rf_i += self.sp_i * self.ref1
        self.I_h1 *= self.decay_syn
        self.I_h1 += torch.matmul(self.wb_hi, self.sp_i)
        self.us_h1 *= self.decay_s
        self.us_h1 = torch.where(self.rf_h1>0., self.zero, self.us_h1 + self.I_h1)
        self.count_i += self.sp_i

        ### Hidden neuron1
        self.sp_h1 = (self.us_h1 >= self.usth).float()
        self.us_h1 *= 1. - self.sp_h1
        self.rf_h1 += self.sp_h1 * self.ref2  
        self.I_h2 *= self.decay_syn
        self.I_h2 += torch.matmul(self.wb_hh, self.sp_h1)
        self.us_h2 *= self.decay_s
        self.us_h2 = torch.where(self.rf_h2>0., self.zero, self.us_h2 + self.I_h2)
        self.count_h1 += self.sp_h1
        
        ### Hidden neuron2
        self.sp_h2 = (self.us_h2 >= self.usth).float()
        self.us_h2 *= 1. - self.sp_h2
        self.rf_h2 += self.sp_h2 * self.ref2  
        self.I_o *= self.decay_syn
        self.I_o += torch.matmul(self.wb_oh, self.sp_h2)
        self.us_o *= self.decay_s
        self.us_o = torch.where(self.rf_o>0., self.zero, self.us_o + self.I_o)
        self.count_h2 += self.sp_h2
        
        ### Prediction neuron
        self.sp_o = (self.us_o >= self.usth).float()
        self.us_o *= 1. - self.sp_o
        self.rf_o += self.sp_o * self.ref2
        self.us_e1 += self.w_E * (self.sp_o - self.sp_l)
        self.us_e2 += self.w_E * (self.sp_l - self.sp_o)
        self.count_o += self.sp_o
        
        ### Error neuron
        self.sp_e1 = (self.us_e1 >= self.usth).float()
        self.us_e1 -= self.sp_e1 * self.usth       
        self.sp_e2 = (self.us_e2 >= self.usth).float()
        self.us_e2 -= self.sp_e2 * self.usth
        
        self.count_e1 += self.sp_e1
        self.count_e2 += self.sp_e2
        
        ### Dendritic potential change by feedback
        self.ud_h1 *= self.decay_d
        self.ud_h1 += torch.matmul(self.gp_h1, self.sp_e1 - self.sp_e2)
        self.ud_h2 *= self.decay_d
        self.ud_h2 += torch.matmul(self.gp_h2, self.sp_e1 - self.sp_e2)
        self.ud_o *= self.decay_d
        self.ud_o += self.w_E * (self.sp_e1 - self.sp_e2)
        
        ### Weight update
        self.w_hi -= self.rl * (self.ud_h1 + 0.0005 * self.lam_hi * -self.w_hi) * self.boxcar(self.I_h1, self.bmax1, self.bmin1) * self.sp_i.transpose(0, 1)
        self.w_hh -= self.rl * (self.ud_h2 + 0.0005 * self.lam_hh * -self.w_hh) * self.boxcar(self.I_h2, self.bmax2, self.bmin2) * self.sp_h1.transpose(0, 1)
        self.w_oh -= self.rl * (self.ud_o + 0.0005 * self.lam_oh * -self.w_oh) * self.boxcar(self.I_o, self.bmax1, self.bmin1) * self.sp_h2.transpose(0, 1)
                                
        self.w_hi = torch.clamp(self.w_hi, min=-1., max=1.)
        self.w_hh = torch.clamp(self.w_hh, min=-1., max=1.)
        self.w_oh = torch.clamp(self.w_oh, min=-1., max=1.)                
  
        ### Lamda update
        self.lam_hi += 0.2 * self.rl * torch.abs(1. - self.w_hi ** 2) * self.boxcar(self.I_h1, self.bmax1, self.bmin1) * self.sp_i.transpose(0, 1)
        self.lam_hh += 0.2 * self.rl * torch.abs(1. - self.w_hh ** 2) * self.boxcar(self.I_h2, self.bmax2, self.bmin2) * self.sp_h1.transpose(0, 1)
        self.lam_oh += 0.2 * self.rl * torch.abs(1. - self.w_oh ** 2) * self.boxcar(self.I_o, self.bmax1, self.bmin1) * self.sp_h2.transpose(0, 1)
        
        ### Refractory
        self.rf_i -= self.dt
        self.rf_h1 -= self.dt
        self.rf_h2 -= self.dt      
        self.rf_o -= self.dt
        F.relu(self.rf_i, inplace=True)
        F.relu(self.rf_h1, inplace=True)
        F.relu(self.rf_h2, inplace=True)
        F.relu(self.rf_o, inplace=True)

    def test_step(self):   
        ### input jump ###
        if self.jump == 1:
            self.tI_h1 *= 0.
            self.tus_h1 *= 0.   

            self.tI_h2 *= 0.
            self.tus_h2 *= 0. 
            
            self.tI_o *= 0.
            self.tus_o *= 0.
            
            self.trf_i *= 0.
            self.trf_h1 *= 0.
            self.trf_h2 *= 0.
            self.trf_o *= 0.
        
            self.jump = 0
        
        ## Neuron firing:
        ### Input neuron
        self.tsp_i = torch.where(self.trf_i>0., self.zero, self.tsp_i)
        self.trf_i += self.tsp_i * self.ref1
        self.tI_h1 *= self.decay_syn
        self.tI_h1 += torch.bmm(self.tw_hi, self.tsp_i)
        self.tus_h1 *= self.decay_s
        self.tus_h1 = torch.where(self.trf_h1>0., self.zero, self.tus_h1 + self.tI_h1)

        ### Hidden neuron1
        self.tsp_h1 = (self.tus_h1 >= self.usth).float()
        self.tus_h1 *= 1. - self.tsp_h1
        self.trf_h1 += self.tsp_h1 * self.ref2  
        self.tI_h2 *= self.decay_syn
        self.tI_h2 += torch.bmm(self.tw_hh, self.tsp_h1)
        self.tus_h2 *= self.decay_s
        self.tus_h2 = torch.where(self.trf_h2>0., self.zero, self.tus_h2 + self.tI_h2)
        
        ### Hidden neuron2
        self.tsp_h2 = (self.tus_h2 >= self.usth).float()
        self.tus_h2 *= 1. - self.tsp_h2
        self.trf_h2 += self.tsp_h2 * self.ref2  
        self.tI_o *= self.decay_syn
        self.tI_o += torch.bmm(self.tw_oh, self.tsp_h2)
        self.tus_o *= self.decay_s
        self.tus_o = torch.where(self.trf_o>0., self.zero, self.tus_o + self.tI_o)
        
        ### Prediction neuron
        self.tsp_o = (self.tus_o >= self.usth).float()
        self.tus_o *= 1. - self.tsp_o
        self.trf_o += self.tsp_o * self.ref2      

        ### Refractory
        self.trf_i -= self.dt
        self.trf_h1 -= self.dt
        self.trf_h2 -= self.dt
        self.trf_o -= self.dt
        F.relu(self.trf_i, inplace=True)
        F.relu(self.trf_h1, inplace=True)
        F.relu(self.trf_h2, inplace=True)
        F.relu(self.trf_o, inplace=True)

def binary_score(Data):
    score = torch.mean((1 - torch.abs(Data)) ** 2).item()
    return score

def test(model, n_i, n_h1, n_h2, n_o, n_test, epoch, acc_step, batch, bi):
    spike_count = torch.cuda.FloatTensor(n_test, n_o, 1).fill_(0)
    
    ### Weight matrix batch
    model.tw_hi = model.w_hi.unsqueeze(0).expand(batch, n_h1, n_i)
    model.tw_hh = model.w_hh.unsqueeze(0).expand(batch, n_h2, n_h1)
    model.tw_oh = model.w_oh.unsqueeze(0).expand(batch, n_o, n_h2)
    
    if bi == 1:
        model.tw_hi = torch.sign(model.tw_hi)
        model.tw_hh = torch.sign(model.tw_hh)
        model.tw_oh = torch.sign(model.tw_oh)        
    
    x = x_test[:n_test, :] * dt
    
    for t_itr in range(int(n_test/batch)):
        model.jump = 1
        x = x_test[t_itr * batch:t_itr * batch + batch, :] * dt
        for t in range(200):
            ### Poisson input            
            model.tsp_i[:, :, 0] = ((x > torch.cuda.FloatTensor(batch, n_i).uniform_()).float())
            model.test_step()
           
            spike_count[t_itr * batch:t_itr * batch + batch, :, :] += model.tsp_o
    
    count = np.sum(y_test[0:n_test].reshape(-1,1) - spike_count.cpu().numpy().argmax(axis=1) == 0)
    return count

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = torch.from_numpy(x_train.reshape(60000, 784).transpose()).float().cuda() + 10
x_test = torch.from_numpy(x_test.reshape(10000, 784)).float().cuda() + 10


## Network simulation:
dt = 1e-3
T = 200
ref2 = 4e-3
n_i, n_h1, n_h2, n_o, n_e1, n_e2, n_l = 784, 500, 500, 10, 10, 10, 10

model = eRBP(batch=500, n_i=n_i, n_h1=n_h1, n_h2=n_h2, n_o=n_o, n_e1=n_e1, n_e2=n_e2, n_l=n_l, dt=1e-3, tau_syn=4e-3, tau_s=1e-3, tau_d=5e-3, 
             ref1=4e-3, ref2=4e-3, usth=1.1, bmin1=-25, bmax1=25, bmin2=-25, bmax2=25, w_E=1, p=0.45, rl=0.001)

start = time.time()
print('@@@@@ start @@@@@')
### Learning
n_train, n_test, epoch, acc_step, batch = 60000, 10000, 25, 10000, 500
step = []
accuracy = []
accuracy_bi = []
bi_score_w_hi = []
bi_score_w_hh = []
bi_score_w_oh = []
timestep = []

for epo in range(epoch):
    for itr in range(n_train):
        model.jump = 1
        x = x_train[:, itr] * dt
        y = torch.cuda.FloatTensor(n_o).fill_(0)
        y[y_train[itr]] = 1/ref2 * dt        
        rand_mat = torch.cuda.FloatTensor(T, n_i).uniform_()

        for t in range(T):
            model.t = t
            model.sp_i[:, 0] = (x > rand_mat[t, :]).float()
            model.sp_l[:, 0] = (y > torch.cuda.FloatTensor(n_o).uniform_()).float()
            model.learning_step()            

        if np.mod(itr, acc_step) == 0:         
            count = test(model, n_i, n_h1, n_h2, n_o, n_test, epoch, acc_step, batch, bi=0)
            count_bi = test(model, n_i, n_h1, n_h2, n_o, n_test, epoch, acc_step, batch, bi=1)

            print("epoch: %02d"%(epo), end='   ')
            print("step: %05d"%(itr), end='   ')
            print("acc:", "%.4f" %(count/n_test), end='   ')
            print("acc_bi :", "%.4f" %(count_bi/n_test), end='   ')
            print("bs_w_hi:", "%.3f" %(binary_score(model.w_hi)), end='   ')
            print("bs_w_hh:", "%.3f" %(binary_score(model.w_hh)), end='   ')
            print("bs_w_oh:", "%.3f" %(binary_score(model.w_oh)), end='   ')
            print("time: ", "%.1f" %(time.time() - start))
            
            step.append(itr + epo * n_train)
            accuracy.append(count/n_test)
            accuracy_bi.append(count_bi/n_test)
            bi_score_w_hi.append(binary_score(model.w_hi))
            bi_score_w_hh.append(binary_score(model.w_hh))
            bi_score_w_oh.append(binary_score(model.w_oh))
            timestep.append(time.time() - start)

            