import torch
import torch.nn as nn
import torch.nn.functional as F
from . import layers

class LeNet5(nn.Module):
    def __init__(self, img_size, n_channels, n_outputs):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(n_channels, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.flat_size = int(img_size/4 - 2)**2 * 16
        self.fc1 = nn.Linear(self.flat_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_outputs)
        
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.flat_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class BayesianLeNet5(nn.Module):
    def __init__(self, img_size, n_channels, n_outputs, prior_scale=1.0, prior_pi=None, reparam='local'):
        super(BayesianLeNet5, self).__init__()
        if reparam=='local':
            Conv2D = layers.Conv2DLocalReparameterization
            Linear = layers.LinearLocalReparameterization
        else:
            Conv2D = layers.Conv2DReparameterization
            Linear = layers.LinearReparameterization
        
        self.n_outputs = n_outputs
        
        self.conv1 = Conv2D(n_channels, 6, 5, stride=1, padding=2, prior_scale=prior_scale, prior_pi=prior_pi)
        self.conv2 = Conv2D(6, 16, 5, stride=1, prior_scale=prior_scale, prior_pi=prior_pi)
        self.flat_size = int(img_size/4 - 2)**2 * 16
        self.fc1 = Linear(self.flat_size, 120, prior_scale=prior_scale, prior_pi=prior_pi)
        self.fc2 = Linear(120, 84, prior_scale=prior_scale, prior_pi=prior_pi)
        self.fc3 = Linear(84, n_outputs, prior_scale=prior_scale, prior_pi=prior_pi)
        
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.flat_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    @property
    def kl(self):
        kl = 0
        for layer in self.modules():
            if hasattr(layer, '_kl'):
                kl += layer._kl
        return kl
    
class BBBC013Net(nn.Module):
    def __init__(self, n_channels, n_outputs):
        super(BBBC013Net, self).__init__()
        self.conv1 = nn.Conv2d(n_channels, 8, 3)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.flat_size = 32*8*8
        self.fc1 = nn.Linear(self.flat_size, n_outputs)
        
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=3, stride=3)
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=3, stride=3)
        x = F.max_pool2d(F.relu(self.conv3(x)), kernel_size=3, stride=3)
        x = x.view(-1, self.flat_size)
        x = self.fc1(x)
        return x
    
class BBBC013BayesNet(nn.Module):
    def __init__(self, n_channels, n_outputs, prior_scale=1.0, prior_pi=None, reparam='local'):
        super(BBBC013BayesNet, self).__init__()
        if reparam=='local':
            Conv2D = layers.Conv2DLocalReparameterization
            Linear = layers.LinearLocalReparameterization
        else:
            Conv2D = layers.Conv2DReparameterization
            Linear = layers.LinearReparameterization
            
        self.n_outputs = n_outputs
        
        self.conv1 = Conv2D(n_channels, 8, 3, prior_scale=prior_scale, prior_pi=prior_pi)
        self.conv2 = Conv2D(8, 16, 3, prior_scale=prior_scale, prior_pi=prior_pi)
        self.conv3 = Conv2D(16, 32, 3, prior_scale=prior_scale, prior_pi=prior_pi)
        self.flat_size = 32*8*8
        self.fc1 = Linear(self.flat_size, n_outputs, prior_scale=prior_scale, prior_pi=prior_pi)
        
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=3, stride=3)
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=3, stride=3)
        x = F.max_pool2d(F.relu(self.conv3(x)), kernel_size=3, stride=3)
        x = x.view(-1, self.flat_size)
        x = self.fc1(x)
        return x
    
    @property
    def kl(self):
        kl = 0
        for layer in self.modules():
            if hasattr(layer, '_kl'):
                kl += layer._kl
        return kl
    
class BayesianMultiScaleCNN(nn.Module):

    def __init__(self, n_outputs, n_channels=3, n_features=1024, prior_scale=1.0, prior_pi=None, reparam='local'):
        super(BayesianMultiScaleCNN, self).__init__()
        if reparam=='local':
            self.Conv2D = layers.Conv2DLocalReparameterization
            self.Linear = layers.LinearLocalReparameterization
        else:
            self.Conv2D = layers.Conv2DReparameterization
            self.Linear = layers.LinearReparameterization
        
        self.n_channels = n_channels
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.prior_scale = prior_scale
        self.prior_pi = prior_pi

        self.pathway1 = self.Pathway(1, 16, 64 )
        self.pathway2 = self.Pathway(2, 16, 32, 2)
        self.pathway3 = self.Pathway(3, 16, 16, 4)
        self.pathway4 = self.Pathway(4, 32, 8,  8)
        self.pathway5 = self.Pathway(5, 32, 4,  16)
        self.pathway6 = self.Pathway(6, 32, 2,  32)

        self.conv = nn.Sequential(
            self.Conv2D(144, n_features, 1, prior_scale=self.prior_scale, prior_pi=self.prior_pi),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(16,20), stride=1)
        )
        
        self.dense = nn.Sequential(
            nn.BatchNorm1d(n_features), 
            self.Linear(n_features, n_outputs, prior_scale=self.prior_scale, prior_pi=self.prior_pi)
        )

    def Pathway(self, pathway_id, depth, post_pool_sz, pre_pool_sz=None):
        pathway = nn.Sequential()
        if pre_pool_sz :
            pathway.add_module('prepool_%i'%(pathway_id), nn.MaxPool2d(pre_pool_sz, pre_pool_sz))
        pathway.add_module('conv0_%i'%(pathway_id), self.Conv2D(self.n_channels, depth, 5, stride=1, padding=2, prior_scale=self.prior_scale, prior_pi=self.prior_pi))
        pathway.add_module('relu0_%i'% (pathway_id), nn.ReLU())
        pathway.add_module('bn0_%i' % (pathway_id), nn.BatchNorm2d(depth))
        pathway.add_module('conv1_%i'%(pathway_id), self.Conv2D(depth, depth, 5, stride=1, padding=2, prior_scale=self.prior_scale, prior_pi=self.prior_pi))
        pathway.add_module('relu1_%i'% (pathway_id), nn.ReLU())
        pathway.add_module('bn1_%i' % (pathway_id), nn.BatchNorm2d(depth))
        pathway.add_module('conv2_%i'%(pathway_id), self.Conv2D(depth, depth, 5, stride=1, padding=2, prior_scale=self.prior_scale, prior_pi=self.prior_pi))
        pathway.add_module('relu2_%i'% (pathway_id), nn.ReLU())
        pathway.add_module('bn2_%i' % (pathway_id), nn.BatchNorm2d(depth))
        pathway.add_module('postpool_%i'%(pathway_id), nn.MaxPool2d(kernel_size=post_pool_sz, stride=post_pool_sz))
        return pathway

    def forward(self, x):
        pathways_output = [self.pathway1(x), 
                           self.pathway2(x), 
                           self.pathway3(x), 
                           self.pathway4(x), 
                           self.pathway5(x), 
                           self.pathway6(x)]
        output = torch.cat(tuple(pathways_output), dim=1)
        output = self.conv(output)
        output = output.view(-1, self.n_features)
        output = self.dense(output)
        return output
    
    @property
    def kl(self):
        kl = 0
        for layer in self.modules():
            if hasattr(layer, '_kl'):
                kl += layer._kl
        return kl
    
import torch.nn as nn

class MultiScaleCNN(nn.Module):

    def __init__(self, n_outputs, n_channels=3, n_features=1024):
        super(MultiScaleCNN, self).__init__()
        self.n_channels = n_channels
        self.n_features = n_features
        self.n_outputs = n_outputs

        self.pathway1 = self.Pathway(1, 16, 64 )
        self.pathway2 = self.Pathway(2, 16, 32, 2)
        self.pathway3 = self.Pathway(3, 16, 16, 4)
        self.pathway4 = self.Pathway(4, 32, 8,  8)
        self.pathway5 = self.Pathway(5, 32, 4,  16)
        self.pathway6 = self.Pathway(6, 32, 2,  32)

        self.conv = nn.Sequential(
            nn.Conv2d(144, n_features, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(16,20), stride=1)
        )
        
        self.dense = nn.Sequential(
            nn.BatchNorm1d(n_features), 
            nn.Linear(n_features, n_outputs)
        )

    def Pathway(self, pathway_id, depth, post_pool_sz, pre_pool_sz=None):
        pathway = nn.Sequential()
        if pre_pool_sz :
            pathway.add_module('prepool_%i'%(pathway_id), nn.MaxPool2d(pre_pool_sz, pre_pool_sz))
        pathway.add_module('conv0_%i'%(pathway_id), nn.Conv2d(self.n_channels, depth, 5, stride=1, padding=2))
        pathway.add_module('relu0_%i'% (pathway_id), nn.ReLU())
        pathway.add_module('bn0_%i' % (pathway_id), nn.BatchNorm2d(depth))
        pathway.add_module('conv1_%i'%(pathway_id), nn.Conv2d(depth, depth, 5, stride=1, padding=2))
        pathway.add_module('relu1_%i'% (pathway_id), nn.ReLU())
        pathway.add_module('bn1_%i' % (pathway_id), nn.BatchNorm2d(depth))
        pathway.add_module('conv2_%i'%(pathway_id), nn.Conv2d(depth, depth, 5, stride=1, padding=2))
        pathway.add_module('relu2_%i'% (pathway_id), nn.ReLU())
        pathway.add_module('bn2_%i' % (pathway_id), nn.BatchNorm2d(depth))
        pathway.add_module('postpool_%i'%(pathway_id), nn.MaxPool2d(kernel_size=post_pool_sz, stride=post_pool_sz))
        return pathway

    def forward(self, x):
        pathways_output = [self.pathway1(x), 
                           self.pathway2(x), 
                           self.pathway3(x), 
                           self.pathway4(x), 
                           self.pathway5(x), 
                           self.pathway6(x)]
        
        output = torch.cat(tuple(pathways_output), dim=1)
        output = self.conv(output)
        output = output.view(-1, self.n_features)
        output = self.dense(output)
        return output