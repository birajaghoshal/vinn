import torch
import numpy as np
import sys
import time
from torch.nn import Parameter
from distributions import Normal, MixtureNormal

def normal_initializer(size, mean=0.0, std=0.1):
    return Parameter(torch.normal(mean=mean*torch.ones(size), std=std))

def mean_field_normal_initializer(size, loc=0.0, ro=-3.0):
    return {"loc": normal_initializer(size, mean=loc), "ro": normal_initializer(size, mean=ro)}

### ------------------EXPERIMENTAL-----------------------
from torch.nn.functional import softplus

def multivariate_normal_initializer(size, loc=0.0):
    return {"loc": normal_initializer(size, mean=loc), "L": L_initializer(size)}

def L_initializer(size):
    n_outputs = size[0]
    n_inputs = size[1]
    kernel_numel = size[2]*size[3]
    L = torch.randn([n_outputs, n_inputs, kernel_numel, kernel_numel])
    for i in range(n_outputs):
        for j in range(n_inputs):
            L[i, j] = torch.potrf(torch.eye(kernel_numel)*torch.normal(torch.tensor(-3.0), torch.tensor(0.1)), upper=False)
            #L[i, j] = torch.randn([kernel_numel, kernel_numel])
            #for k in range(kernel_numel):
            #    L[i, j, k, k] = L[i, j, k, k].abs()
            #L[i, j] = torch.tril(L[i, j])
    return torch.nn.Parameter(L)

### ---------------------------------------------------------


def default_prior(scale=1., size=1):
    return Normal(loc=torch.zeros(size), scale=scale*torch.ones(size))

def scale_mixture_prior(scale=[10, 0.01], pi=[.5, .5]):
    return MixtureNormal(loc=Parameter(torch.zeros(len(scale)), requires_grad=False),
                         scale=Parameter(torch.tensor(scale), requires_grad=False), 
                         pi=Parameter(torch.tensor(pi), requires_grad=False))

def predict(model, data_loader, n_samples, n_classes, device, output_activation=torch.nn.Softmax(dim=1)):
    model.eval()
    
    batch_size = data_loader.batch_size
    len_dataset = len(data_loader.dataset)

    total = n_samples * len_dataset
    current = 0
    
    start = time.time()
    
    sys.stdout.write("[" + " "*50 + "] {:6.2f}% | {} min {} s".format(0, 0, 0))
    
    samples = np.empty((n_samples, len_dataset, n_classes))
    for n in range(n_samples):
        with torch.no_grad():
            output = torch.empty((len_dataset, n_classes))
            for i, (input, _) in enumerate(data_loader):
                input = input.to(device)
                output_mb = output_activation(model(input))
                output[i*batch_size:min(len_dataset, (i+1)*batch_size), :] = output_mb

                current += len(output_mb)
                perc = float(current)/total*100

                sys.stdout.write("\r[" + "-" * int(perc/2) + " " * (50-int(perc/2)) + "] {:6.2f}% | {} min {} s"\
                                 .format(perc, int((time.time()-start)/60), int((time.time()-start)%60)))

            samples[n, :, :] = output.data.cpu().numpy()

    outputs = np.mean(samples, axis=0)
    uncertainty = np.array([output_variance(samples[:,i,:], output) for i, output in enumerate(outputs)])
    variances = uncertainty[:, 0]
    labels = np.argmax(outputs, 1)
    
    return labels, variances, outputs, samples

def output_variance(p, p_mean, top=True):
    aleatoric = np.mean(p - np.square(p), axis=0)
    epistemic = np.mean(np.square(p - np.tile(p_mean, (len(p), 1))), axis=0)
    if top:
        aleatoric = aleatoric[np.argmax(p_mean)]
        epistemic = epistemic[np.argmax(p_mean)]
    return aleatoric + epistemic, aleatoric, epistemic

def mc_kl_divergence(p, q, n_samples=1):
    kl = 0
    for _ in range(n_samples):
        sample = p.rsample()
        kl += p.log_prob(sample) - q.log_prob(sample)
    return kl / n_samples

def kl_divergence(p, q):
    var_ratio = (p.scale / q.scale.to(p.loc.device)).pow(2)
    t1 = ((p.loc - q.loc.to(p.loc.device)) / q.scale.to(p.loc.device)).pow(2)
    return 0.5 * (var_ratio + t1 - 1 - var_ratio.log())
   



