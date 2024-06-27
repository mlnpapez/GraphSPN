import math
import torch
import torch.nn as nn
import qmcpy

from torch.distributions import Categorical
from typing import Callable, Optional, List
from abc import abstractmethod
from utils import *

# the continuous mixture model and the convolutional decoder based on the following implementation: https://github.com/AlCorreia/cm-tpm


class ResBlockOord(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(),
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)

def mlp_decoder(nd_in: int, nd_out: int, nl: int, batch_norm: bool, final_act: Optional[any]=None):
    nd_hid = torch.arange(nd_in, nd_out, (nd_out - nd_in) / nl, dtype=torch.int)
    decoder = nn.Sequential()
    for i in range(len(nd_hid) - 1):
        decoder.append(nn.Linear(nd_hid[i], nd_hid[i + 1]))
        decoder.append(nn.LeakyReLU())
        if batch_norm:
            decoder.append(nn.BatchNorm1d(nd_hid[i + 1]))
        # decoder.append(nn.Dropout(p=0.2))
    decoder.append(nn.Linear(nd_hid[-1], nd_out))
    if final_act is not None:
        decoder.append(final_act)
    return decoder

def conv_decoder(latent_dim: int, nf: int, batch_norm: Optional[bool]=True, final_act: Optional[nn.Module]=None, bias: Optional[bool]=False, learn_std: Optional[bool]=False, resblock: Optional[bool]=False, out_channels: Optional[int]=None):
    decoder = nn.Sequential()   
    decoder.append(nn.Unflatten(1, (latent_dim, 1, 1)))
    decoder.append(nn.ConvTranspose2d(latent_dim, nf * 4, 2, 2, 0, bias=bias))
    if batch_norm:
        decoder.append(nn.BatchNorm2d(nf * 4))
    decoder.append(nn.LeakyReLU())
    if resblock:
        decoder.append(ResBlockOord(nf * 4))

    decoder.append(nn.ConvTranspose2d(nf * 4, nf * 2, 2, 2, 0, bias=bias))
    if batch_norm:
        decoder.append(nn.BatchNorm2d(nf * 2))
    decoder.append(nn.LeakyReLU())

    if resblock:
        decoder.append(ResBlockOord(nf * 2))

    decoder.append(nn.ConvTranspose2d(nf * 2, nf, 2, 2, 0, bias=bias))
    if batch_norm:
        decoder.append(nn.BatchNorm2d(nf))
    decoder.append(nn.LeakyReLU())

    if resblock:
        decoder.append(ResBlockOord(nf))

    decoder.append(nn.ConvTranspose2d(nf, out_channels, 2, 2, 4, 1, bias=bias))

    if final_act is not None:
        decoder.append(final_act)

    return decoder

class GaussianQMCSampler:
    def __init__(self, nd: int, num_samples: int, mean: Optional[float]=0.0, covariance: Optional[float]=1.0):
        self.nd = nd
        self.num_samples = num_samples
        self.mean = mean
        self.covariance = covariance

    def __call__(self, seed: Optional[int]=None, dtype=torch.float32):
        if seed is None:
            seed = torch.randint(10000, (1,)).item()
        latt = qmcpy.Lattice(dimension=self.nd, randomize=True, seed=seed)
        dist = qmcpy.Gaussian(sampler=latt, mean=self.mean, covariance=self.covariance)
        z = torch.from_numpy(dist.gen_samples(self.num_samples))
        log_w = torch.full(size=(self.num_samples,), fill_value=math.log(1 / self.num_samples))
        return z.type(dtype), log_w.type(dtype)


class CategoricalDecoder(nn.Module):
    def __init__(self, nd_node: int, nk_node: int, nk_edge: int, net_back: nn.Module, net_node: nn.Module, net_edge: nn.Module, device: Optional[str]='cuda'):
        super(CategoricalDecoder, self).__init__()
        self.nd_node = nd_node
        self.nd_edge = nd_node**2
        self.nk_node = nk_node
        self.nk_edge = nk_edge
        self.net_back = net_back
        self.net_node = net_node
        self.net_edge = net_edge
        self.device = device

    def network(self, z):
        h_back = self.net_back(z)
        h_node, h_edge = torch.chunk(h_back, 2, 1)
        h_node = self.net_node(h_node)
        h_edge = self.net_edge(h_edge)
        return h_node, h_edge

    def forward(self, x_node: torch.Tensor, x_edge: torch.Tensor, z: torch.Tensor, num_chunks: Optional[int]=None):
        z_chunks = tuple([z]) if num_chunks is None else z.chunk(num_chunks, dim=0)
        log_prob = []
        x_edge = x_edge.view(-1, self.nd_edge)
        for z_chunk in z_chunks:
            logit_node, logit_edge = self.network(z_chunk.to(self.device))
            logit_edge = torch.movedim(logit_edge, 1, -1)
            logit_node = logit_node.view(-1, self.nd_node, self.nk_node)
            logit_edge = logit_edge.view(-1, self.nd_edge, self.nk_edge)
            log_prob_node = Categorical(logits=logit_node).log_prob(x_node.to(self.device).unsqueeze(1).float())
            log_prob_edge = Categorical(logits=logit_edge).log_prob(x_edge.to(self.device).unsqueeze(1).float())
            log_prob.append(torch.cat([log_prob_node, log_prob_edge], dim=2))
        log_prob = torch.cat(log_prob, dim=1)
        return log_prob.sum(dim=2)

    def sample(self, z: torch.Tensor, k: torch.Tensor):
        logit_node, logit_edge = self.network(z[k].to(self.device))
        logit_edge = torch.movedim(logit_edge, 1, -1)
        logit_node = logit_node.view(-1, self.nd_node, self.nk_node)
        logit_edge = logit_edge.view(-1, self.nd_edge, self.nk_edge)
        x_node = Categorical(logits=logit_node).sample()
        x_edge = Categorical(logits=logit_edge).sample()
        x_edge = x_edge.view(-1, self.nd_node, self.nd_node)
        return x_node, x_edge


class ContinuousMixture(nn.Module):
    def __init__(self, decoder: nn.Module, sampler: Callable, num_chunks: Optional[int]=2, device: Optional[str]='cuda'):
        super(ContinuousMixture, self).__init__()
        self.decoder = decoder
        self.sampler = sampler
        self.num_chunks = num_chunks
        self.device = device

    def forward(self, x_node: torch.Tensor, x_edge: torch.Tensor, z: Optional[torch.Tensor]=None, log_w: Optional[torch.Tensor]=None, seed: Optional[int]=None):
        if z is None:
            z, log_w = self.sampler(seed=seed)
        log_prob = self.decoder(x_node, x_edge, z, self.num_chunks)
        log_prob = torch.logsumexp(log_prob + log_w.to(self.device).unsqueeze(0), dim=1)
        return log_prob

    def sample(self, num_samples: int):
        z, log_w = self.sampler()
        k = Categorical(logits=log_w).sample([num_samples])
        x_node, x_edge = self.decoder.sample(z, k)
        return x_node, x_edge


class GraphSPNBackCore(nn.Module):
    def __init__(self, nd_n: int, nk_n: int, nk_e: int, nz: int, nl: int, nb: int, nc: int, atom_list: List[int], device: Optional[str]='cuda'):
        super().__init__()
        nd_e = nd_n**2
        self.nd_nodes = nd_n
        self.nd_edges = nd_e
        nd_back = 512

        decoder = CategoricalDecoder(
            nd_n,
            nk_n,
            nk_e,
            mlp_decoder(nz,        nd_back,   nl, True),
            mlp_decoder(nd_back//2, nd_n*nk_n, 4, True),
            mlp_decoder(nd_back//2, nd_e*nk_e, 4, True)
            # conv_decoder(nd_back//2, 32, out_channels=4, resblock=True, bias=True, learn_std=True)
        )

        self.network = ContinuousMixture(
            decoder=decoder,
            sampler=GaussianQMCSampler(nz, nb),
            num_chunks=nc,
            device=device
        )

        self.atom_list = atom_list

        self.device = device
        self.to(device)

    @abstractmethod
    def forward(self, xx, aa):
        pass

    def logpdf(self, x):
        return self(x).mean()

    def sample(self, num_samples):
        x, a = self.network.sample(num_samples)
        x = x.to(torch.int).cpu()
        a = a.to(torch.int).cpu()
        return create_mols(x, a, self.atom_list)


class GraphSPNBackNone(GraphSPNBackCore):
    def __init__(self, nd_n: int, nk_n: int, nk_e: int, nz: int, nl: int, nb: int, nc: int, atom_list: List[int], device: Optional[str]='cuda'):
        super().__init__(nd_n, nk_n, nk_e, nz, nl, nb, nc, atom_list, device)

    def forward(self, x):
        xx = x['x']
        aa = x['a']
        return self.network(xx, aa)


MODELS = {
    'graphspn_back_none': GraphSPNBackNone,
}
