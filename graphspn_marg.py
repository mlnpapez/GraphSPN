import math
from numpy import arange
import torch
import torch.nn as nn
import itertools

from abc import abstractmethod
from einsum import Graph, EinsumNetwork, ExponentialFamilyArray
from torch.distributions import Poisson
from utils import *


def marginalize(network, nd_nodes, num_empty, num_full):
    with torch.no_grad():
        if num_empty > 0:
            mx = torch.zeros(nd_nodes,           dtype=torch.bool)
            ma = torch.zeros(nd_nodes, nd_nodes, dtype=torch.bool)
            mx[num_full:   ] = True
            ma[num_full:, :] = True
            ma[:, num_full:] = True
            m = torch.cat((mx.unsqueeze(1), ma), dim=1)
            marginalization_idx = torch.arange(nd_nodes+nd_nodes**2)[m.view(-1)]

            network.set_marginalization_idx(marginalization_idx)
        else:
            network.set_marginalization_idx(None)


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

    def forward(self, x):
        l = []
        c = torch.count_nonzero(x['x'] == len(self.atom_list), dim=1)
        for num_empty in torch.unique(c):
            num_full = self.nd_nodes-num_empty.item()
            xx = x['x'][c == num_empty]
            aa = x['a'][c == num_empty]
            marginalize(self.network, self.nd_nodes, num_empty, num_full)

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
            marginalize(self.network, self.nd_nodes, num_empty, num_full)

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
        for pi in itertools.permutations(range(num_full), num_full):
            with torch.no_grad():
                pi = torch.cat((torch.tensor(pi), torch.arange(num_full, self.nd_nodes)))
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
            with torch.no_grad():
                pi = torch.cat((pi, torch.arange(num_full-1, self.nd_nodes)))
                xx = xx[:, pi]
                aa = aa[:, pi, :]
                aa = aa[:, :, pi]
            z = torch.cat((xx.unsqueeze(2), aa), dim=2)
            z = z.view(-1, self.nd_nodes + self.nd_edges).to(self.device)
            l[:, i] = self.network(z).squeeze()
        return torch.logsumexp(l, dim=1) - torch.log(torch.tensor(self.num_perms))


class GraphSPNMargSort(GraphSPNMargCore):
    def __init__(self, nd_n, nk_n, nk_e, ns, ni, nl, nr, atom_list, device='cuda'):
        super().__init__(nd_n, nk_n, nk_e, ns, ni, nl, nr, atom_list, device)

    def _forward(self, xx, aa, num_full):
        with torch.no_grad():
            xx, pi = xx.sort(dim=1)
            pi = torch.cat((pi[:, :num_full], torch.arange(num_full, self.nd_nodes).repeat(len(xx), 1)), dim=1)
            for i, p in enumerate(pi):
                aa[i, :, :] = aa[i, p, :]
                aa[i, :, :] = aa[i, :, p]
        z = torch.cat((xx.unsqueeze(2), aa), dim=2)
        z = z.view(-1, self.nd_nodes + self.nd_edges).to(self.device)
        return self.network(z)


class GraphSPNMargkAry(nn.Module):
    def __init__(self, nd_n, nk_n, nk_e, ns, ni, nl, nr, arity, atom_list, device='cuda'):
        super().__init__()
        self.nd_nodes = nd_n
        self.nd_edges = nd_n**2
        self.arity = arity

        nd = arity + arity**2
        nk = max(nk_n, nk_e)
        self.nk_nodes = nk_n
        self.nk_edges = nk_e

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

    def forward(self, x):
        o = []
        c = torch.count_nonzero(x['x'] == len(self.atom_list), dim=1)
        for num_empty in torch.unique(c):
            num_full = self.nd_nodes-num_empty.item()
            xx = x['x'][c == num_empty]
            aa = x['a'][c == num_empty]
            self.network.set_marginalization_idx(None) # TODO: propagate to the rest
            if num_full >= self.arity:
                n = math.comb(num_full, self.arity)
                l = torch.zeros(len(xx), n, device=self.device)
                for i, pi in enumerate(itertools.combinations(range(num_full), self.arity)):
                    with torch.no_grad():
                        _x = xx[:, pi]
                        _a = aa[:, pi, :]
                        _a = _a[:, :, pi]
                    z = torch.cat((_x.unsqueeze(2), _a), dim=2)
                    z = z.view(-1, self.arity + self.arity**2).to(self.device)
                    l[:, i] = self.network(z).squeeze()
            else:
                n = 1
                l = torch.zeros(len(xx), n, device=self.device)
                marginalize(self.network, self.arity, num_empty, num_full)
                with torch.no_grad():
                    pi = torch.arange(self.arity)
                    _x = xx[:, pi]
                    _a = aa[:, pi, :]
                    _a = _a[:, :, pi]
                z = torch.cat((_x.unsqueeze(2), _a), dim=2)
                z = z.view(-1, self.arity + self.arity**2).to(self.device)
                l[:, 0] = self.network(z).squeeze()

            o.append(torch.logsumexp(l, dim=1) - torch.log(torch.tensor(n)))

        num_empty, _ = c.sort()
        num_full = self.nd_nodes - num_empty
        d = Poisson(self.rate.exp())

        return d.log_prob(num_full.to(self.device)) + torch.cat(o)

    def logpdf(self, x):
        return self(x).mean()

    def sample(self, num_samples):
        mols = []
        smls = []
        d = Poisson(self.rate.exp())
        c = self.nd_nodes - d.sample((num_samples, )).clamp(0, self.nd_nodes).to(torch.int)
        for num_empty, num_samples in zip(*torch.unique(c, return_counts=True)):
            num_full = self.nd_nodes-num_empty.item()
            self.network.set_marginalization_idx(None) # TODO: propagate to the rest
            if num_full >= self.arity:
                n = math.comb(num_full, self.arity)
                s = torch.randint(n, (1,))
                sub_graph = list(itertools.combinations(range(num_full), self.arity))[s]
            else:
                marginalize(self.network, self.nd_nodes, num_empty, num_full)
                sub_graph = torch.arange(num_full)

            x = torch.randint(0, self.nk_nodes, (num_samples, num_full),           dtype=torch.int)
            a = torch.randint(0, self.nk_edges, (num_samples, num_full, num_full), dtype=torch.int)
            z = self.network.sample(num_samples).to(torch.int).cpu()

            z = z.view(-1, self.arity, self.arity+1)
            for i in range(min(self.arity, num_full)):
                x[:, sub_graph[i]] = z[:, i, 0]
                for j in range(min(self.arity, num_full)):
                    a[:, sub_graph[i], sub_graph[j]] = z[:, i, j+1]

            _mols, _smls = create_mols(x, a, self.atom_list)
            mols.extend(_mols)
            smls.extend(_smls)

        return mols, smls


MODELS = {
    'graphspn_marg_none': GraphSPNMargNone,
    'graphspn_marg_full': GraphSPNMargFull,
    'graphspn_marg_rand': GraphSPNMargRand,
    'graphspn_marg_sort': GraphSPNMargSort,
    'graphspn_marg_kary': GraphSPNMargkAry,
}
