from __future__ import print_function
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri
#%matplotlib inline
from matplotlib import colors
from IPython import display
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
from plotly.graph_objs import Scatter, Figure, Layout

import numpy as np
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import _pickle as cPickle
import gzip, numpy
#print('package imported!')


# Load the dataset
f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f,encoding='latin1')
f.close()
del valid_set
del test_set
train_img = torch.from_numpy(train_set[0].reshape(50000, 1, 28, 28))
del train_set
#print('data loaded!')


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape
    def forward(self, input):
        return input.view(self.shape)

class Generator(nn.Module):
    def __init__(self, ninput, ngf):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
        # input is Z, going into a convolution
            nn.ConvTranspose2d(ninput, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
#             # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     1, 4, 2, 3, bias=False),
            nn.Tanh()
#             state size. (ngf) x 28 x 28            
        )

    def forward(self, input):
#         if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
#             output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
#         else:
        output = self.main(input)
        return output


class Discriminator(nn.Module):
    def __init__(self, ninput, ndf):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
           #input is 28 x 28
            nn.Conv2d(ninput, ndf * 2, 3, 2, 3, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
#         if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
#             output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
#         else:
        output = self.main(input)

        return output.view(-1, 1).squeeze(1)


class GAN():
    def __init__(self, data, ninput, nlatent, ngf, ndf, nbatch, dis_steps, gen_steps):
        self.ninput = ninput
        self.nlatent = nlatent
        self.nbatch = nbatch
        self.dis_steps = dis_steps
        self.gen_steps = gen_steps
        
        self.real_data = data
        self.discriminator = Discriminator(ninput, ndf)
        self.generator = Generator(nlatent, ngf)
        
        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)        
        
        if cuda:
            self.generator = self.generator.cuda()
            self.discriminator = self.discriminator.cuda()
            #self.real_data = self.real_data.cuda()
        self.init_lr = lr
       # self.gen_optim = optim.RMSprop(self.generator.parameters(), lr=lr)
        #self.dis_optim = optim.RMSprop(self.discriminator.parameters(), lr=lr)
        self.dis_optim = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.gen_optim = optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))

        self.score_real = []
        self.score_fake = []
        self.loss_avg_gen = []
        self.loss_avg_dis = []             
    def adjust_lr(self, optimizer, iter_step):
        lr = self.init_lr * (0.9 ** (iter_step // 20))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr     
    def train(self, iter_step):        

#         self.adjust_lr(self.gen_optim, iter_step)
#         self.adjust_lr(self.dis_optim, iter_step)
                
        # train discriminator        
        self.generator.eval()
        self.discriminator.train()
        for j in range(self.dis_steps):
            self.dis_optim.zero_grad()
            self.gen_optim.zero_grad()
            
            # clamp parameters to a cube, only for WGAN
            for p in self.discriminator.parameters():
                p.data.clamp_(-0.05, 0.05)
            
            epsilon = torch.randn(self.nbatch, self.nlatent,1,1)
            ind = torch.from_numpy(np.random.choice(50000,self.nbatch))
            real = self.real_data[ind]
            if cuda:
                real = real.cuda()
                epsilon = epsilon.cuda()
            real = Variable(real)        
            epsilon = Variable(epsilon, requires_grad=False)

            fake = self.generator(epsilon)
            real_scores = self.discriminator(real)
            fake_scores = self.discriminator(fake)
            
            # GAN Loss
            #loss_dis = torch.mean(-torch.log(real_scores)-torch.log(1-fake_scores))
            
            # WGAN Loss
            loss_dis = torch.mean(fake_scores)-torch.mean(real_scores)
            
         #   if loss_dis.data[0] > -0.1:
            loss_dis.backward()
            self.dis_optim.step()
            real_score = torch.mean(real_scores)
            fake_score = torch.mean(fake_scores)
        
        self.score_real.append(real_score.data[0])
        self.score_fake.append(fake_score.data[0])
        
        # train generator        
        self.generator.train()
        self.discriminator.eval()
        for j in range(self.gen_steps):
            self.gen_optim.zero_grad()
            self.dis_optim.zero_grad()
            epsilon = torch.randn(self.nbatch, self.nlatent,1,1)
            if cuda:
                epsilon = epsilon.cuda()
            epsilon = Variable(epsilon, requires_grad=False)

            fake = self.generator(epsilon)
            fake_score = self.discriminator(fake)
            
            # GAN Loss
           # loss_gen = torch.mean(-torch.log(fake_score))
            
            # WGAN Loss
            loss_gen = torch.mean(-fake_score)
            
            
            loss_gen.backward()
            self.gen_optim.step()
#         print(loss_gen.data[0])
#         print(loss_dis.data[0])
        self.loss_avg_gen.append(loss_gen.data[0])
        self.loss_avg_dis.append(loss_dis.data[0])
#         print(self.loss_avg_gen)
#         print(self.loss_avg_dis)
        self.generator.eval()
        if iter_step % 1000 ==999:
        #    self.plot_distributions(iter_step)
        #    plt.savefig('GAN')
            torch.save(self.score_real, 'score_real15')
            torch.save(self.score_fake, 'score_fake15')
            torch.save(self.loss_avg_gen, 'loss_gen15')
            torch.save(self.loss_avg_dis, 'loss_dis15')
            torch.save(self.discriminator.state_dict(), 'GAN_MINST_DISC15_%s'%int(iter_step/1000))
            torch.save(self.generator.state_dict(), 'GAN_MINST_GEN15_%s'%int(iter_step/1000))
           
            
            
    def plot_distributions(self, nsteps):
        fig = plt.figure(figsize = (15,5))
        font = {'size':20}	
        ax1 = fig.add_subplot(121)
        ax1.imshow(test_out1[0].squeeze(1).data.cpu().numpy().reshape(28,28))
#         plt.show()
        
        ax2 = fig.add_subplot(132)
        ax2.plot(np.arange(nsteps+1), np.array(self.loss_avg_dis), label='dis loss')
        ax2.plot(np.arange(nsteps+1), np.array(self.loss_avg_gen), label='gen loss')
        ax2.set_title('GAN Loss',font)
        ax2.set_xlabel('Iterations',font)
        ax2.set_ylabel('Loss',font)
        ax2.legend()
        
        ax3 = fig.add_subplot(133)
        ax3.plot(np.arange(nsteps+1), np.array(self.score_real), label='real score')
        ax3.plot(np.arange(nsteps+1), np.array(self.score_fake), label='fake score')
        ax3.set_title('GAN Scores',font)
        ax3.set_xlabel('Iterations',font)
        ax3.set_ylabel('Score',font)
        ax3.legend()
#         plt.tight_layout()
        

nc = 1
nz = 100
ngf = 32
ndf = 64
niter = 1000 
nbatch = 64
lr = 0.0002
dis_steps = 1
gen_steps = 2
cuda = True
#print('initialization done!')


model = GAN(
    data=train_img,
    nbatch=nbatch,
    nlatent=nz,
    ninput=nc,
    ngf = ngf,
    ndf = ndf,
    dis_steps=dis_steps,
    gen_steps=gen_steps,
)
#print('model initialized')

for i in range(niter):
    model.train(i)
#    print('training done!')

