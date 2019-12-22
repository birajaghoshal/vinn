import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import softplus
from distributions import Normal

class VIModule(nn.Module):
   
    def __init__(self, 
                 weight_posterior,
                 weight_prior,
                 bias_posterior,
                 bias_prior,
                 kl_divergence_fn):
        
        super(VIModule, self).__init__()       
        
        self.weight_loc = weight_posterior['loc']
        self.weight_ro = weight_posterior['ro']
        self.weight_prior = weight_prior
        
        if bias_posterior is None:
            self.bias_loc = None
            self.bias_ro = None
            self.bias_prior = None
        else:
            self.bias_loc = bias_posterior['loc']
            self.bias_ro = bias_posterior['ro']
            self.bias_prior = bias_prior

        self.kl_divergence = kl_divergence_fn
    
    @property
    def _kl(self):
        kl = torch.sum(self.kl_divergence(Normal(self.weight_loc, softplus(self.weight_ro)), self.weight_prior))
        if self.bias_loc is not None:
            kl += torch.sum(self.kl_divergence(Normal(self.bias_loc, softplus(self.bias_ro)), self.bias_prior))
        return kl
    
    @staticmethod
    def rsample(loc, scale):
        eps = torch.empty(loc.shape, dtype=loc.dtype, device=loc.device).normal_()
        return loc + eps * scale
    
    def extra_repr(self):
        s = "weight: {}, bias: {}".format(list(self.weight_loc.size()), list(self.bias_loc.size()))     
        return s
        
class ReparameterizationLayer(VIModule):
    
    def __init__(self, 
                 weight_posterior,
                 weight_prior,
                 bias_posterior,
                 bias_prior,
                 kl_divergence_fn,
                 layer_fn,
                 **kwargs):
    
        super(ReparameterizationLayer, self).__init__(weight_posterior,
                                                      weight_prior,
                                                      bias_posterior,
                                                      bias_prior,
                                                      kl_divergence_fn)
        self.layer_fn = layer_fn
        self.kwargs = kwargs
        
    def forward(self, input):    
        if self.bias_loc is None:
            return self.layer_fn(input, self.rsample(self.weight_loc, softplus(self.weight_ro)), **self.kwargs)
        return self.layer_fn(input, self.rsample(self.weight_loc, softplus(self.weight_ro)), self.rsample(self.bias_loc, softplus(self.bias_ro)), **self.kwargs)

class LocalReparameterizationLayer(VIModule):
    
    def __init__(self, 
                 weight_posterior,
                 weight_prior,
                 bias_posterior,
                 bias_prior,
                 kl_divergence_fn,
                 layer_fn,
                 **kwargs):
    
        super(LocalReparameterizationLayer, self).__init__(weight_posterior,
                                                           weight_prior,
                                                           bias_posterior,
                                                           bias_prior,
                                                           kl_divergence_fn)
        self.layer_fn = layer_fn
        self.kwargs = kwargs
    
    def forward(self, input):
        if self.bias_loc is None:
            output_loc = self.layer_fn(input, self.weight_loc, **self.kwargs)
            output_scale = torch.sqrt(1e-9 + self.layer_fn(input.pow(2), softplus(self.weight_ro)**2, **self.kwargs))
            return self.rsample(output_loc, output_scale)
        output_loc = self.layer_fn(input, self.weight_loc, self.bias_loc, **self.kwargs)
        output_scale = torch.sqrt(1e-9 + self.layer_fn(input.pow(2), softplus(self.weight_ro)**2, softplus(self.bias_ro)**2, **self.kwargs))
        return self.rsample(output_loc, output_scale)
    
class LinearReparameterization(ReparameterizationLayer):
    
    def __init__(self, 
                 weight_posterior,
                 weight_prior,
                 bias_posterior,
                 bias_prior,
                 kl_divergence_fn):
    
        super(LinearReparameterization, self).__init__(weight_posterior,
                                                       weight_prior,
                                                       bias_posterior,
                                                       bias_prior,
                                                       kl_divergence_fn,
                                                       F.linear)
        
class LinearLocalReparameterization(LocalReparameterizationLayer):
    
    def __init__(self, 
                 weight_posterior,
                 weight_prior,
                 bias_posterior,
                 bias_prior,
                 kl_divergence_fn):
    
        super(LinearLocalReparameterization, self).__init__(weight_posterior,
                                                            weight_prior,
                                                            bias_posterior,
                                                            bias_prior,
                                                            kl_divergence_fn,
                                                            F.linear)
        
class Conv2DReparameterization(ReparameterizationLayer):
    
    def __init__(self, 
                 weight_posterior,
                 weight_prior,
                 bias_posterior,
                 bias_prior,
                 kl_divergence_fn,
                 stride=1, 
                 padding=0, 
                 dilation=1, 
                 groups=1):
        
        super(Conv2DReparameterization, self).__init__(weight_posterior,
                                                       weight_prior,
                                                       bias_posterior,
                                                       bias_prior,
                                                       kl_divergence_fn,
                                                       F.conv2d,
                                                       stride=stride, 
                                                       padding=padding, 
                                                       dilation=dilation, 
                                                       groups=groups)

### ---------------------EXPERIMENTAL--------------------------        
 
from torch.distributions import MultivariateNormal
    
class VIModuleExperimental(nn.Module):
   
    def __init__(self, 
                 weight_posterior,
                 weight_prior,
                 bias_posterior,
                 bias_prior,
                 kl_divergence_fn):
        
        super(VIModuleExperimental, self).__init__()       
        
        self.weight_loc = weight_posterior['loc']
        self.weight_L = weight_posterior['L']
        self.weight_prior = weight_prior
        
        if bias_posterior is None:
            self.bias_loc = None
            self.bias_ro = None
            self.bias_prior = None
        else:
            self.bias_loc = bias_posterior['loc']
            self.bias_ro = bias_posterior['ro']
            self.bias_prior = bias_prior

        self.kl_divergence = kl_divergence_fn
    
    @property
    def _kl(self):
        #kl = torch.sum(self.kl_divergence(MultivariateNormal(self.weight_loc, self.weight_L), self.weight_prior))
        if self.bias_loc is not None:
            kl += torch.sum(self.kl_divergence(Normal(self.bias_loc, softplus(self.bias_ro)), self.bias_prior))
        return kl
    
    @staticmethod
    def rsample(loc, scale):
        eps = torch.empty(loc.shape, dtype=loc.dtype, device=loc.device).normal_()
        return loc + eps * scale
    
    def extra_repr(self):
        s = "weight: {}, bias: {}".format(list(self.weight_loc.size()), list(self.bias_loc.size()))  
        return s
    
class Conv2DExperimental(VIModuleExperimental):
    
    def __init__(self, 
                 weight_posterior,
                 weight_prior,
                 bias_posterior,
                 bias_prior,
                 kl_divergence_fn,
                 **kwargs):
    
        super(Conv2DExperimental, self).__init__(weight_posterior,
                                                      weight_prior,
                                                      bias_posterior,
                                                      bias_prior,
                                                      kl_divergence_fn)
        self.kwargs = kwargs
        
    def forward(self, input):    
        if self.bias_loc is None:
            return F.conv2d(input, self.weight_rsample(), **self.kwargs)
        return F.conv2d(input, self.weight_rsample(), self.rsample(self.bias_loc, softplus(self.bias_ro)), **self.kwargs)
    
    def weight_rsample(self):
        out_channels = self.weight_loc.size(0)
        in_channels = self.weight_loc.size(1)
        kernel_size = self.weight_loc.size(2)
        return torch.cat([torch.cat([MultivariateNormal(loc=self.weight_loc[j, i].view(-1),
                                                     scale_tril=self.diag_softplus(self.weight_L[j, i])).rsample()\
                                  .view(1, kernel_size, kernel_size) for i in range(in_channels)])\
                       .view(1, in_channels, kernel_size, kernel_size) for j in range(out_channels)])
    
    def diag_softplus(self, L):
        for i in range(L.size(0)):
            L[i, i] = softplus(L[i, i])
        return L

### -----------------------------------------------------------------------------
        
class Conv2DLocalReparameterization(LocalReparameterizationLayer):
    
    def __init__(self, 
                 weight_posterior,
                 weight_prior,
                 bias_posterior,
                 bias_prior,
                 kl_divergence_fn,
                 stride=1, 
                 padding=0, 
                 dilation=1, 
                 groups=1):
    
        super(Conv2DLocalReparameterization, self).__init__(weight_posterior,
                                                            weight_prior,
                                                            bias_posterior,
                                                            bias_prior,
                                                            kl_divergence_fn,
                                                            F.conv2d,
                                                            stride=stride, 
                                                            padding=padding, 
                                                            dilation=dilation, 
                                                            groups=groups) 