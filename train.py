import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
import torch.utils.data as data
from torch.autograd import Variable
import numpy as np
from PIL import Image

from skimage.measure import compare_psnr as psnr
import scipy.io as sio

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor
#dtype = torch.FloatTensor
modelType = 'IDM'

class DBlock(nn.Module):
    def __init__(self):
        super(DBlock,self).__init__()
        
        self.conv1 = nn.Conv2d(64,48,3,padding=1)
        self.lrelu1 = nn.LeakyReLU(0.05)
        self.conv2 = nn.Conv2d(48,32,3,padding=1)
        self.lrelu2 = nn.LeakyReLU(0.05)
        self.conv3 = nn.Conv2d(32,64,3,padding=1)
        self.lrelu3 = nn.LeakyReLU(0.05)
        
        self.conv4 = nn.Conv2d(48,64,3,padding=1)
        self.lrelu4 = nn.LeakyReLU(0.05)
        self.conv5 = nn.Conv2d(64,48,3,padding=1)
        self.lrelu5 = nn.LeakyReLU(0.05)
        self.conv6 = nn.Conv2d(48,80,3,padding=1)
        self.lrelu6 = nn.LeakyReLU(0.05)
        
        self.conv7 = nn.Conv2d(160,64,1,padding=0)
        self.lrelu7 = nn.LeakyReLU(0.05)
        
    def forward(self,x1):
        x2 = self.conv1(x1)
        x2 = self.lrelu1(x2)
        x2 = self.conv2(x2)
        x2 = self.lrelu2(x2)
        x2 = self.conv3(x2)
        x2 = self.lrelu3(x2)
        
        x3 = x2[:,0:48,:,:]
        x4 = x2[:,48:64,:,:]
        
        x5 = self.conv4(x3)
        x5 = self.lrelu4(x5)
        x5 = self.conv5(x5)
        x5 = self.lrelu5(x5)
        x5 = self.conv6(x5)
        x5 = self.lrelu6(x5)
        
        x6 = torch.cat((x1,x4,x5),1)
        x7 = self.conv7(x6)
        x7 = self.lrelu7(x7)
        
        return x7
        
class IDN(nn.Module):
    def __init__(self):
        super(IDN, self).__init__()
        
        self.FBlock = nn.Sequential()
        self.FBlock.add_module("conv_0", nn.Conv2d(4,32,3,padding=1))
        self.FBlock.add_module("LeakyReLU_0", nn.LeakyReLU(0.05))
        self.FBlock.add_module("conv_1", nn.Conv2d(32,64,3,padding=1))
        self.FBlock.add_module("LeakyReLU_1", nn.LeakyReLU(0.05))
        
        self.DBlock1 = DBlock()
        self.DBlock2 = DBlock()
        self.DBlock3 = DBlock()
        
        self.RBlock = nn.Sequential()
        self.RBlock.add_module("deconv",nn.ConvTranspose2d(64,4,4,stride = 4,padding=0,output_padding=0))
        
    def forward(self,x1):
        x2 = self.FBlock(x1)
        x3 = self.DBlock1(x2)
        x4 = self.DBlock2(x3)
        x5 = self.DBlock3(x4)
        x6 = self.RBlock(x5)
        
        return x6

 # get training patches
def get_batch(train_data,bs,trainSize=0): 
    
    gt = train_data['gt'][...]
    #print(gt.shape)
    gt = np.transpose(gt,(0,3,1,2))
    pan = train_data['pan'][...]  #### Pan image N*H*W
    ms_lr = train_data['ms'][...]
    ms_lr = np.transpose(ms_lr,(0,3,1,2))
    lms   = train_data['lms'][...]
    lms = np.transpose(lms,(0,3,1,2))
   
    gt = np.array(gt,dtype = np.float32) / 1.  ### normalization, WorldView L = 11
    pan = np.array(pan, dtype = np.float32) /1.
    ms_lr = np.array(ms_lr, dtype = np.float32) / 1.
    lms  = np.array(lms, dtype = np.float32) /1.
    

    
    N = gt.shape[0]
    if(trainSize!=0):
        N = trainSize
    batch_index = np.random.randint(0,N,size = bs)
    
    gt_batch = gt[batch_index,:,:,:]
    pan_batch = pan[batch_index,:,:]
    ms_lr_batch = ms_lr[batch_index,:,:,:]
    lms_batch  = lms[batch_index,:,:,:]
    
    #pan_hp_batch = get_edge(pan_batch)
    #pan_hp_batch = pan_hp_batch[:,:,:,np.newaxis] # expand to N*H*W*1
    pan_batch = pan_batch[:,np.newaxis,:,:] # expand to N*H*W*1
    
    #ms_hp_batch = get_edge(ms_lr_batch)
    
    
    return gt_batch, lms_batch, pan_batch, ms_lr_batch

def test():

    gt,lms,pan,ms = get_batch(train_data,test_batch_size)
    
    
    netInput = Variable(torch.from_numpy(ms)).type(dtype)
    #print(ms.shape)
    netLabel = Variable(torch.from_numpy(gt)).type(dtype)
    lmsVar = Variable(torch.from_numpy(lms)).type(dtype)
    res = net(netInput)
    output = res+lmsVar

    loss = mse(output,netLabel)

    #loss = mse(netIn,netOut)

    netOutput_np = output.cpu().data.numpy()
    netLabel_np = netLabel.cpu().data.numpy()
    psnrValue = psnr(netLabel_np,netOutput_np)


    print('psnr %.2f'%(psnrValue))

if __name__ == "__main__"
    train_batch_size = 64 # training batch size
    test_batch_size = 1  # validation batch size
    #image_size = 64      # patch size
    iteration = 50000 # total number of iterations to use.
    LR = 0.0001
    checkpoint_iter = 0
    model_directory = './models' # directory to save trained model to.
    train_data_name = './training_data/train.mat'  # training data
    test_data_name  = './training_data/validation.mat'   # validation data

    net = IDN()
    net = nn.DataParallel(net,device_ids=[0,1,2,3]).cuda()

    train_data = sio.loadmat(train_data_name)
    test_data = sio.loadmat(test_data_name)

    mse = torch.nn.MSELoss().type(dtype)

    if checkpoint_iter != 0:
        net.load_state_dict(torch.load('weights/saved_'+modelType+'_epoch%d.pkl'%(checkpoint_iter)))

    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    
    for j in range(iteration):
        
        #========closure=========
        gt,lms,pan,ms = get_batch(train_data,train_batch_size)
        optimizer.zero_grad()
    
        netInput = Variable(torch.from_numpy(ms)).type(dtype)
        #print(ms.shape)
        netLabel = Variable(torch.from_numpy(gt)).type(dtype)
        lmsVar = Variable(torch.from_numpy(lms)).type(dtype)
        res = net(netInput)
        output = res+lmsVar
        loss = mse(output,netLabel)
        loss.backward()
        
        optimizer.step()
        print ('Iteration %05d loss %.8f' % (j+checkpoint_iter,loss.data[0]), '\r', end='')
    #========================
    
        if j % 200 == 0:
            model_path = 'weights/saved_'+modelType+'_epoch%d.pkl'%(j+checkpoint_iter)
            torch.save(net.state_dict(), model_path)
            print("")
            test()