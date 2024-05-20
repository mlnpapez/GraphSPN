import math
import torch
import torch.nn as nn
import itertools

from abc import abstractmethod
from einsum import Graph, EinsumNetwork, ExponentialFamilyArray
from torch.distributions import Poisson
from utils import *
from tqdm import tqdm


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

            x, a = unflatt_graph(z, self.nd_nodes, num_full)

            _mols, _smls = create_mols(x, a, self.atom_list)
            mols.extend(_mols)
            smls.extend(_smls)

        return mols, smls


class GraphSPNMargNone(GraphSPNMargCore):
    def __init__(self, nd_n, nk_n, nk_e, ns, ni, nl, nr, atom_list, device='cuda'):
        super().__init__(nd_n, nk_n, nk_e, ns, ni, nl, nr, atom_list, device)

    def _forward(self, xx, aa, num_full):
        z = flatten_graph(xx, aa)
        return self.network(z.to(self.device))


class GraphSPNMargFull(GraphSPNMargCore):
    def __init__(self, nd_n, nk_n, nk_e, ns, ni, nl, nr, atom_list, device='cuda'):
        super().__init__(nd_n, nk_n, nk_e, ns, ni, nl, nr, atom_list, device)

    def _forward(self, xx, aa, num_full):
        n = torch.tensor(math.factorial(num_full))
        l = torch.zeros(len(xx), device=self.device)
        for pi in tqdm(itertools.permutations(range(num_full), num_full), leave=False):
            with torch.no_grad():
                pi = torch.cat((torch.tensor(pi), torch.arange(num_full, self.nd_nodes)))
                px, pa = permute_graph(xx, aa, pi)
            z = flatten_graph(px, pa)
            l += (torch.exp(self.network(z.to(self.device)).squeeze() - torch.log(n)))
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
                px, pa = permute_graph(xx, aa, pi)
            z = flatten_graph(px, pa)
            l[:, i] = self.network(z.to(self.device)).squeeze()
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
        z = flatten_graph(xx, aa)
        return self.network(z.to(self.device))


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
            self.network.set_marginalization_idx(None) # TODO: propagate to the rest

            xx = x['x'][c == num_empty]
            aa = x['a'][c == num_empty]

            if num_full >= self.arity:
                n = math.comb(num_full, self.arity)
                l = torch.zeros(len(xx), n, device=self.device)
                for i, pi in enumerate(itertools.combinations(range(num_full), self.arity)):
                    with torch.no_grad():
                        px, pa = permute_graph(xx, aa, pi)
                    z = flatten_graph(px, pa)
                    l[:, i] = self.network(z.to(self.device)).squeeze()
            else:
                n = 1
                l = torch.zeros(len(xx), n, device=self.device)
                marginalize(self.network, self.arity, num_empty, num_full)
                with torch.no_grad():
                    pi = torch.arange(self.arity)
                    px, pa = permute_graph(xx, aa, pi)
                z = flatten_graph(px, pa)
                l[:, 0] = self.network(z.to(self.device)).squeeze()

            o.append(torch.logsumexp(l, dim=1) - torch.log(torch.tensor(n)))

        num_empty, _ = c.sort()
        num_full = self.nd_nodes - num_empty
        d = Poisson(self.rate.exp())

        return d.log_prob(num_full.to(self.device)) + torch.cat(o)

    def logpdf(self, x):
        return self(x).mean()

    def sample(self, num_samples):
        num_sub_graphs = 20
        mols = []
        smls = []
        d = Poisson(self.rate.exp())
        c = self.nd_nodes - d.sample((num_samples, )).clamp(0, self.nd_nodes).to(torch.int)
        for num_empty, num_samples in zip(*torch.unique(c, return_counts=True)):
            num_full = self.nd_nodes-num_empty.item()
            self.network.set_marginalization_idx(None) # TODO: propagate to the rest

            x = torch.randint(0, self.nk_nodes, (num_samples, num_full),           dtype=torch.int)
            a = torch.randint(0, self.nk_edges, (num_samples, num_full, num_full), dtype=torch.int)

            if num_full >= self.arity:
                n = math.comb(num_full, self.arity)
                sub_graphs = list(itertools.combinations(range(num_full), self.arity))
                sub_graphs = [sub_graphs[i] for i in torch.randint(n, (num_sub_graphs,)).tolist()]

                z = self.network.sample(num_sub_graphs*num_samples).to(torch.int).cpu()
                z = z.view(num_sub_graphs, num_samples, self.arity, self.arity+1)

                for g in range(num_sub_graphs):
                    for i in range(min(self.arity, num_full)):
                        x[:, sub_graphs[g][i]] = z[g, :, i, 0]
                        for j in range(min(self.arity, num_full)):
                            a[:, sub_graphs[g][i], sub_graphs[g][j]] = z[g, :, i, j+1]
            else:
                marginalize(self.network, self.nd_nodes, num_empty, num_full)

                z = self.network.sample(num_samples).to(torch.int).cpu()

                x[:, 0:num_full], a[:, 0:num_full, 0:num_full] = unflatt_graph(z, self.arity, num_full)

            _mols, _smls = create_mols(x, a, self.atom_list)
            mols.extend(_mols)
            smls.extend(_smls)

        return mols, smls


class GraphSPNMargFree(nn.Module):
    def __init__(self, nd_n, nk_n, nk_e, ns, ni, nl, nr, atom_list, device='cuda'):
        super().__init__()
        self.nd_nodes = nd_n

        nd = nd_n + 1
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

    def forward(self, x):
        o = []
        c = torch.count_nonzero(x['x'] == len(self.atom_list), dim=1)
        for num_empty in torch.unique(c):
            num_full = self.nd_nodes-num_empty.item()
            self.network.set_marginalization_idx(None) # TODO: propagate to the rest

            m = c == num_empty
            n = m.sum()
            xx = x['x'][m][:, 0:num_full]
            aa = x['a'][m][:, 0:num_full, 0:num_full]
            xx = xx.reshape(n*num_full)
            aa = aa.reshape(n*num_full, num_full)
            aa = torch.cat((aa, torch.zeros(n*num_full, num_empty)), dim=1)

            with torch.no_grad():
                if num_empty > 0:
                    mask = torch.zeros(self.nd_nodes+1, dtype=torch.bool)
                    mask[num_full:] = True
                    marginalization_idx = torch.arange(self.nd_nodes+1)[mask]

                    self.network.set_marginalization_idx(marginalization_idx)
                else:
                    self.network.set_marginalization_idx(None)

            z = torch.cat((xx.unsqueeze(1), aa), dim=1)
            o.append(self.network(z.to(self.device)))

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
        c = self.nd_nodes - d.sample((num_samples, )).clamp(1, self.nd_nodes).to(torch.int)
        for num_empty, num_samples in zip(*torch.unique(c, return_counts=True)):
            num_full = self.nd_nodes-num_empty.item()
            if num_empty > 0:
                mask = torch.zeros(self.nd_nodes+1, dtype=torch.bool)
                mask[num_full:] = True
                marginalization_idx = torch.arange(self.nd_nodes+1)[mask]

                self.network.set_marginalization_idx(marginalization_idx)
            else:
                self.network.set_marginalization_idx(None)

            z = self.network.sample(num_samples*num_full).to(torch.int).cpu()

            x = z[:, 0 ].view(-1, num_full)
            a = z[:, 1:].view(-1, num_full, self.nd_nodes)
            a = a[:, :, 0:num_full]

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
    'graphspn_marg_free': GraphSPNMargFree,
}
