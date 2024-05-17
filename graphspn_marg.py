import math
import torch
import torch.nn as nn
import itertools

from abc import abstractmethod
from einsum import Graph, EinsumNetwork, ExponentialFamilyArray
from torch.distributions import Poisson
from utils import *

class GraphSPNMargCore(nn.Module):
    def __init__(self, nd_n, nk_n, nk_e, ns, ni, nl, nr, atom_list, device='cuda'):
        super().__init__()
        nd_e = nd_n**2
        self.nd_nodes = nd_n
        self.nd_edges = nd_e

        nd = nd_n + nd_e
        nk = max(nk_n, nk_e)

        graph = Graph.random_binary_trees(nd, nl, nr)

        args = EinsumNetwork.Args(
            num_var=nd,
            num_input_distributions=ni,
            num_sums=ns,
            exponential_family=ExponentialFamilyArray.CategoricalArray,
            exponential_family_args={'K': nk},
            use_em=False)

        self.network = EinsumNetwork.EinsumNetwork(graph, args)
        self.network.initialize()

        self.rate = nn.Parameter(torch.randn(1, device=device), requires_grad=True)

        self.atom_list = atom_list

        self.device = device
        self.to(device)

    @abstractmethod
    def _forward(self, xx, aa, num_full):
        pass

    def _marginalize(self, num_empty, num_full):
        with torch.no_grad():
            if num_empty > 0:
                mx = torch.zeros(self.nd_nodes,                dtype=torch.bool)
                ma = torch.zeros(self.nd_nodes, self.nd_nodes, dtype=torch.bool)
                mx[num_full:   ] = True
                ma[num_full:, :] = True
                ma[:, num_full:] = True
                m = torch.cat((mx.unsqueeze(1), ma), dim=1)
                marginalization_idx = torch.arange(self.nd_nodes+self.nd_edges)[m.view(-1)]

                self.network.set_marginalization_idx(marginalization_idx)
            else:
                self.network.set_marginalization_idx(None)

    def forward(self, x):
        l = []
        c = torch.count_nonzero(x['x'] == len(self.atom_list), dim=1)
        for num_empty in torch.unique(c):
            num_full = self.nd_nodes-num_empty.item()
            xx = x['x'][c == num_empty]
            aa = x['a'][c == num_empty]
            self._marginalize(num_empty, num_full)

            l.append(self._forward(xx, aa, num_full))

        num_empty, _ = c.sort()
        num_full = self.nd_nodes - num_empty
        d = Poisson(self.rate.exp())

        return d.log_prob(num_full.to(self.device)) + torch.cat(l)

    def logpdf(self, x):
        return self(x).mean()

    def sample(self, num_samples):
        mols = []
        smls = []
        d = Poisson(self.rate.exp())
        c = self.nd_nodes - d.sample((num_samples, )).clamp(0, self.nd_nodes).to(torch.int)
        for num_empty, num_samples in zip(*torch.unique(c, return_counts=True)):
            num_full = self.nd_nodes-num_empty.item()
            self._marginalize(num_empty, num_full)

            z = self.network.sample(num_samples).to(torch.int).cpu()

            z = z.view(-1, self.nd_nodes, self.nd_nodes+1)
            x = z[:, 0:num_full, 0 ]
            a = z[:, 0:num_full, 1:num_full+1]

            _mols, _smls = create_mols(x, a, self.atom_list)
            mols.extend(_mols)
            smls.extend(_smls)

        return mols, smls


class GraphSPNMargNone(GraphSPNMargCore):
    def __init__(self, nd_n, nk_n, nk_e, ns, ni, nl, nr, atom_list, device='cuda'):
        super().__init__(nd_n, nk_n, nk_e, ns, ni, nl, nr, atom_list, device)

    def _forward(self, xx, aa, num_full):
        z = torch.cat((xx.unsqueeze(2), aa), dim=2)
        z = z.view(-1, self.nd_nodes + self.nd_edges).to(self.device)
        return self.network(z)


class GraphSPNMargFull(GraphSPNMargCore):
    def __init__(self, nd_n, nk_n, nk_e, ns, ni, nl, nr, atom_list, device='cuda'):
        super().__init__(nd_n, nk_n, nk_e, ns, ni, nl, nr, atom_list, device)

    def _forward(self, xx, aa, num_full):
        n = torch.tensor(math.factorial(num_full))
        l = torch.zeros(len(xx), device=self.device)
        for i, pi in enumerate(itertools.permutations(range(num_full), num_full)):
            r = torch.arange(num_full, self.nd_nodes)
            pi = torch.cat((torch.tensor(pi), r))
            xx = xx[:, pi]
            aa = aa[:, pi, :]
            aa = aa[:, :, pi]
            z = torch.cat((xx.unsqueeze(2), aa), dim=2)
            z = z.view(-1, self.nd_nodes + self.nd_edges).to(self.device)
            l += (torch.exp(self.network(z).squeeze() - torch.log(n)))
        return torch.log(l)


class GraphSPNMargRand(GraphSPNMargCore):
    def __init__(self, nd_n, nk_n, nk_e, ns, ni, nl, nr, np, atom_list, device='cuda'):
        super().__init__(nd_n, nk_n, nk_e, ns, ni, nl, nr, atom_list, device)

        self.num_perms = np
        self.permutations = {i:torch.stack([torch.randperm(i) for _ in range(min(np, math.factorial(i)))]) for i in range(nd_n)}

    def _forward(self, xx, aa, num_full):
        l = torch.zeros(len(xx), min(self.num_perms, math.factorial(num_full)), device=self.device)
        for i, pi in enumerate(self.permutations[num_full-1]):
            pi = torch.cat((pi, torch.arange(num_full-1, self.nd_nodes)))
            xx = xx[:, pi]
            aa = aa[:, pi, :]
            aa = aa[:, :, pi]
            z = torch.cat((xx.unsqueeze(2), aa), dim=2)
            z = z.view(-1, self.nd_nodes + self.nd_edges).to(self.device)
            l[:, i] = self.network(z).squeeze()
        return torch.logsumexp(l, dim=1) - torch.log(torch.tensor(self.num_perms))



MODELS = {
    'graphspn_marg_none': GraphSPNMargNone,
    'graphspn_marg_full': GraphSPNMargFull,
    'graphspn_marg_rand': GraphSPNMargRand,
}
