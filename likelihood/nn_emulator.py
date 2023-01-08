import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from tqdm import tqdm
import numpy as np
import h5py as h5
from torchvision import models
from torchinfo import summary


class Affine(nn.Module):
    def __init__(self):
        super(Affine, self).__init__()

        self.gain = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x * self.gain + self.bias

class ResBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super(ResBlock, self).__init__()
        
        if in_size != out_size:
            self.skip = nn.Linear(in_size, out_size, bias=False)
        else:
            self.skip = nn.Identity()

        self.layer1 = nn.Linear(in_size, out_size)
        self.layer2 = nn.Linear(out_size, out_size)

        self.norm1 = Affine()
        self.norm2 = Affine()

        # self.act1 = nn.PReLU()
        # self.act2 = nn.PReLU()
        self.act1 = nn.Tanh()
        self.act2 = nn.Tanh()

    def forward(self, x):
        xskip = self.skip(x)

        o1 = self.layer1(self.act1(self.norm1(x))) / np.sqrt(10)
        o2 = self.layer2(self.act2(self.norm2(o1))) / np.sqrt(10) + xskip

        return o2

###KZ begin: below is a dummy example that can run####

# class Bin_Split(nn.Module):
#   def __init__(self, in_size, out_size, n_bin):
#     super(Bin_Split, self).__init__()
#     self.n_bin = n_bin
#     if in_size != out_size:
#         self.skip = nn.Linear(in_size, out_size, bias=False)
#     else:
#         self.skip = nn.Identity()

#     self.layer1 = nn.Linear(in_size, out_size)
#     self.layer2 = nn.Linear(out_size, out_size)

#     self.norm1 = Affine()
#     self.norm2 = Affine()

#     self.act1 = nn.PReLU()
#     self.act2 = nn.PReLU()

#   def forward(self, x):
#     xskip = self.skip(x)

#     o1 = self.layer1(self.act1(self.norm1(x))) / np.sqrt(10)
#     o2 = self.layer2(self.act2(self.norm2(o1))) / np.sqrt(10) + xskip
#     return [o2 for _ in range(self.n_bin)]

###KZ end

class Bin_Split(nn.Module):
  def __init__(self, in_size, out_size, n_bin):
    super(Bin_Split, self).__init__()
    self.n_bin = n_bin

    # # KZv1 start: the modulelist below works, as Linearly connected
    # l_in     = nn.Linear(in_size, 128)
    # l_hidden = nn.Linear(128, 128)
    # l_out    = nn.Linear(128, out_size)
    # act      = nn.PReLU()
    # drop     = nn.Dropout(self.dropout)
    # l = [l_in, act, drop, l_hidden, act, drop, l_hidden, act, drop, l_hidden, act, drop, l_hidden, act, drop, l_out, Affine()]
    # #KZv1 end linear ex end

    # # KZv2 start: same as origional ResBlock for each bin. Maybe too complicated for bin_split
    # l_in     = nn.Linear(in_size, 128)
    # l_out    = nn.Linear(1024, out_size)
    # act      = nn.PReLU()
    # drop     = nn.Dropout(self.dropout)
    # l = [l_in, ResBlock(128, 256),drop, ResBlock(256, 256),drop, ResBlock(256, 256),drop, 
    #     ResBlock(256, 512),drop,ResBlock(512, 512),drop,ResBlock(512, 512),drop,
    #     ResBlock(512, 1024),drop,ResBlock(1024, 1024),drop,ResBlock(1024, 1024),drop,ResBlock(1024, 1024),drop,
    #     ResBlock(1024, 1024), Affine(), act, l_out, Affine()]
    # # KZv2 end:

    # KZv3 start: simplified
    l_in     = nn.Linear(in_size, 32)
    l_out    = nn.Linear(64, out_size)
    act      = nn.PReLU()
    drop     = nn.Dropout(self.dropout)
    l = [l_in, ResBlock(32, 64),drop, ResBlock(64, 64),drop, ResBlock(64, 64),drop, 
         ResBlock(64, 64),  Affine(), act, l_out, Affine()]
    # KZv2 end:

    self.module_list = nn.ModuleList(l)

  def forward(self, x):
    out = []
    for i in range(self.n_bin):
        tmp = x
        for f in self.module_list:
            tmp = f(tmp)
        out.append(tmp)
    return out

    #old version, should be the same?
  # def forward(self, x):
  #   for f in self.module_list:
  #       x = f(x)
  #   return [x for _ in range(self.n_bin)]

    
class NNEmulator:
    def __init__(self, N_DIM, OUTPUT_DIM, dv_fid, dv_std, cov_inv, dv_max, device, model='resnet_small', optim=None):

        torch.set_default_dtype(torch.float32)
        self.N_DIM = N_DIM
        self.model = model
        self.optim = optim
        self.device = device
        self.trained = False
        self.model_type = model
        
        self.dv_fid  = torch.Tensor(dv_fid)
        self.dv_std  = torch.Tensor(dv_std)
        self.cov_inv = torch.Tensor(cov_inv)        
        self.dv_max  = torch.Tensor(dv_max)

        self.output_dim = OUTPUT_DIM

        self.n_bin        = 30 #Number of Tomographic bin
        self.theta_bin    = 26
        self.dropout      = 0.5
        self.learningrate = 1e-4
       
        
        if model == 'simply_connected':
            print("Using simply connected NN...")
            print("model output_dim: ", OUTPUT_DIM)
            self.model = nn.Sequential(
                                nn.Linear(N_DIM, 512),
                                nn.ReLU(),
                                nn.Dropout(self.dropout),
                                nn.Linear(512, 512),
                                nn.ReLU(),
                                nn.Dropout(self.dropout),
                                nn.Linear(512, 512),
                                nn.ReLU(),
                                nn.Dropout(self.dropout),
                                nn.Linear(512, 512),
                                nn.ReLU(),
                                nn.Dropout(self.dropout),
                                nn.Linear(512, 512), #additional laryer
                                nn.ReLU(), #additional laryer
                                nn.Dropout(self.dropout), #additional laryer
                                nn.Linear(512, OUTPUT_DIM),
                                Affine()
                                )            
        elif(model=='resnet'):
            print("Using resnet model...")
            print("model output_dim: ", OUTPUT_DIM)
            self.model = nn.Sequential(
                    nn.Linear(N_DIM, 128),
                    ResBlock(128, 256),
                    nn.Dropout(self.dropout),
                    ResBlock(256, 256),
                    nn.Dropout(self.dropout),
                    ResBlock(256, 256),
                    nn.Dropout(self.dropout),
                    ResBlock(256, 512),
                    nn.Dropout(self.dropout),
                    ResBlock(512, 512),
                    nn.Dropout(self.dropout),
                    ResBlock(512, 512),
                    nn.Dropout(self.dropout),
                    ResBlock(512, 1024),
                    nn.Dropout(self.dropout),
                    ResBlock(1024, 1024),
                    nn.Dropout(self.dropout),
                    ResBlock(1024, 1024),
                    nn.Dropout(self.dropout),
                    ResBlock(1024, 1024),
                    nn.Dropout(self.dropout),
                    ResBlock(1024, 1024),
                    Affine(),
                    nn.PReLU(),
                    nn.Linear(1024, OUTPUT_DIM),
                    Affine()
                )
        elif(model=='resnet_small'):
            print("Using resnet_samll model...")
            print("model output_dim: ", OUTPUT_DIM)
            self.model = nn.Sequential(
                nn.Linear(N_DIM, 128),
                ResBlock(128, 256),
                nn.Dropout(0.3),
                ResBlock(256, 512),
                nn.Dropout(0.3),
                ResBlock(512, 2048),
                nn.Dropout(0.3),
                nn.PReLU(),
                nn.Linear(2048, OUTPUT_DIM),
                Affine()
                )

        elif(model=='bin_split'):
            assert self.theta_bin == OUTPUT_DIM / self.n_bin, "please make sure output_dim = n_bin * theta_bin "

            print("Using bin_split NN...")

            self.model = Bin_Split(N_DIM, self.theta_bin,  self.n_bin)

        ###
        self.model.to(self.device)
        self.model.to(torch.float32)
        if self.optim is None:
            self.optim = torch.optim.Adam(self.model.parameters(), lr=self.learningrate)
        
    def train(self, X, y, X_validation, y_validation, test_split=None, batch_size=32, n_epochs=100):
        if not self.trained:
            self.X_mean = torch.Tensor(X.mean(axis=0, keepdims=True)).float()
            self.X_std  = torch.Tensor(X.std(axis=0, keepdims=True)).float()
    
        epoch_range = tqdm(range(n_epochs))

        losses_train = []
        losses_vali = []
        loss = 100.

        tmp_dv_max       = self.dv_max.to(self.device).float()
        tmp_dv_std       = self.dv_std.to(self.device).float()
        tmp_cov_inv      = self.cov_inv.to(self.device).float()
        tmp_X_mean       = self.X_mean.to(self.device).float()
        tmp_X_std        = self.X_std.to(self.device).float()
        tmp_X_validation = X_validation.to(self.device).float()
        tmp_Y_validation = y_validation.to(self.device).float()

        X_train     = ((X.float() - self.X_mean)/self.X_std)
        y_train     = y
        trainset    = torch.utils.data.TensorDataset(X_train, y_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=1)
    
        for _ in epoch_range:
            for i, data in enumerate(trainloader):    
                X       = data[0].to(self.device).float()                  # This is input
                Y_batch = data[1].to(self.device).float() * tmp_dv_max     # This is label

                if self.model_type =='bin_split':
                    tmp = torch.Tensor().to(self.device)
                    for i in range(self.n_bin):
                        #print("testing", self.model(X)[i], i)
                        tmp = torch.cat((tmp,self.model(X)[i]), dim=1)
                        # print('testing', tmp.size())
                    Y_pred  = tmp * tmp_dv_max
                else:
                    #print(self.model(X).size())
                    Y_pred  = self.model.train()(X) * tmp_dv_max

                loss = torch.mean(torch.diagonal( torch.matmul( torch.matmul((Y_batch - Y_pred),tmp_cov_inv), torch.t(Y_batch - Y_pred)) ) )

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            losses_train.append(loss.cpu().detach().numpy())
            ###Validation loss
            with torch.no_grad():
                if self.model_type =='bin_split':
                    tmp = torch.Tensor().to(self.device)
                    for i in range(self.n_bin):
                        tmp = torch.cat((tmp,self.model.eval()((tmp_X_validation - tmp_X_mean)/tmp_X_std)[i]), dim=1)
                    Y_pred = tmp * tmp_dv_max
                    print("Model summary:", summary(self.model)) #or models.vgg16()
                    quit()
                else:
                    ###KZ: question, what's the difference between using eval or not??
                    Y_pred = self.model.eval()((tmp_X_validation - tmp_X_mean)/tmp_X_std) * tmp_dv_max
                    #print("Model summary:", summary(self.model)) #or models.vgg16()
                    #quit()

            loss_vali = torch.mean(torch.diagonal( torch.matmul( torch.matmul((tmp_Y_validation - Y_pred),tmp_cov_inv), torch.t(tmp_Y_validation - Y_pred)) ) )
 
            losses_vali.append(loss_vali.cpu().detach().numpy())
               
            epoch_range.set_description('Loss: {0}, Loss_validation: {1}'.format(loss, loss_vali))

            if loss_vali < 0.3:
                continue

        #print("Model summary:", summary(self.model))
        torch.set_printoptions(precision=15)

        np.savetxt("losses.txt", [losses_train,losses_vali], fmt='%s')
        #save last dv from validation just for plotting
        np.savetxt("test_dv.txt", np.array( [tmp_Y_validation.cpu().detach().numpy().astype(np.float64)[-1], Y_pred.cpu().detach().numpy().astype(np.float64)[-1] ]), fmt='%s')
        self.trained = True

    def predict(self, X):
        assert self.trained, "The emulator needs to be trained first before predicting"

        with torch.no_grad():
            X_mean = self.X_mean.clone().detach().to(self.device).float()
            X_std  = self.X_std.clone().detach().to(self.device).float()

            y_pred = self.model.eval()((X.to(self.device) - X_mean) / X_std).float().cpu() * self.dv_max
            
        return y_pred.float().numpy()

    def save(self, filename):
        torch.save(self.model, filename)
        with h5.File(filename + '.h5', 'w') as f:
            f['X_mean'] = self.X_mean
            f['X_std']  = self.X_std
            f['dv_fid'] = self.dv_fid
            f['dv_std'] = self.dv_std
            f['dv_max'] = self.dv_max
        
    def load(self, filename, map_location):
        self.trained = True
        self.model = torch.load(filename, map_location)
        with h5.File(filename + '.h5', 'r') as f:
            self.X_mean = torch.Tensor(f['X_mean'][:]).float()
            self.X_std  = torch.Tensor(f['X_std'][:]).float()
            self.dv_fid = torch.Tensor(f['dv_fid'][:]).float()
            self.dv_std = torch.Tensor(f['dv_std'][:]).float()
            self.dv_max = torch.Tensor(f['dv_max'][:]).float()


