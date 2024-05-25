import math
import torch
import torch.nn as nn
import itertools

from abc import abstractmethod
from einsum import Graph, EinsumNetwork, ExponentialFamilyArray
from utils import *
from tqdm import tqdm


class GraphSPNZeroCore(nn.Module):
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

        self.atom_list = atom_list

        self.device = device
        self.to(device)

    @abstractmethod
    def forward(self, xx, aa):
        pass

    def logpdf(self, x):
        return self(x).mean()

    def sample(self, num_samples):
        z = self.network.sample(num_samples).to(torch.int).cpu()
        x, a = unflatt_graph(z, self.nd_nodes, self.nd_nodes)
        return create_mols(x, a, self.atom_list)


class GraphSPNZeroNone(GraphSPNZeroCore):
    def __init__(self, nd_n, nk_n, nk_e, ns, ni, nl, nr, atom_list, device='cuda'):
        super().__init__(nd_n, nk_n, nk_e, ns, ni, nl, nr, atom_list, device)

    def forward(self, x):
        xx = x['x']
        aa = x['a']
        z = flatten_graph(xx, aa)
        return self.network(z.to(self.device))


class GraphSPNZeroFull(GraphSPNZeroCore):
    def __init__(self, nd_n, nk_n, nk_e, ns, ni, nl, nr, atom_list, device='cuda'):
        super().__init__(nd_n, nk_n, nk_e, ns, ni, nl, nr, atom_list, device)

    def forward(self, x):
        xx = x['x']
        aa = x['a']
        n = torch.tensor(math.factorial(self.nd_nodes))
        l = torch.zeros(len(xx), device=self.device)
        for pi in tqdm(itertools.permutations(range(self.nd_nodes), self.nd_nodes), leave=False):
            with torch.no_grad():
                px, pa = permute_graph(xx, aa, pi)
            z = flatten_graph(px, pa)
            l += (torch.exp(self.network(z.to(self.device)).squeeze() - torch.log(n)))
        return torch.log(l)


class GraphSPNZeroRand(GraphSPNZeroCore):
    def __init__(self, nd_n, nk_n, nk_e, ns, ni, nl, nr, np, atom_list, device='cuda'):
        super().__init__(nd_n, nk_n, nk_e, ns, ni, nl, nr, atom_list, device)

        self.num_perms = np
        self.permutations = torch.stack([torch.randperm(nd_n) for _ in range(min(np, math.factorial(nd_n)))])

    def forward(self, x):
        xx = x['x']
        aa = x['a']
        # l = torch.zeros(len(xx), min(self.num_perms, math.factorial(self.nd_nodes)), device=self.device)
        # for i, pi in enumerate(self.permutations):
        # # permutations = torch.stack([torch.randperm(self.nd_nodes) for _ in range(self.num_perms)])
        # # for i, pi in enumerate(permutations):
        #     with torch.no_grad():
        #         px, pa = permute_graph(xx, aa, pi)
        #     z = flatten_graph(px, pa)
        #     l[:, i] = self.network(z.to(self.device)).squeeze()
        # return torch.logsumexp(l, dim=1) - torch.log(torch.tensor(self.num_perms))
        for i in range(len(xx)):
            num_full = torch.sum(xx[i, :] != len(self.atom_list))
            pi = torch.cat((torch.randperm(num_full), torch.arange(num_full, self.nd_nodes)))
            xx[i, :] = xx[i, pi]
            aa[i, :, :] = aa[i, pi, :]
            aa[i, :, :] = aa[i, :, pi]

        z = flatten_graph(xx, aa)
        return self.network(z.to(self.device))

class GraphSPNZeroSort(GraphSPNZeroCore):
    def __init__(self, nd_n, nk_n, nk_e, ns, ni, nl, nr, atom_list, device='cuda'):
        super().__init__(nd_n, nk_n, nk_e, ns, ni, nl, nr, atom_list, device)

    def forward(self, x):
        xx = x['x']
        aa = x['a']
        with torch.no_grad():
            # xx, pi = xx.sort(dim=1)
            # for i, p in enumerate(pi):
            #     aa[i, :, :] = aa[i, p, :]
            #     aa[i, :, :] = aa[i, :, p]
            xs = torch.zeros_like(xx)
            for i in range(len(xx)):
                num_full = torch.sum(xx[i, :] != len(self.atom_list))
                for j in range(self.nd_nodes):
                    if (j < num_full) and (xx[i, j] < len(self.atom_list)):
                        xs[i, j] = self.atom_list[xx[i, j]]
                    else:
                        xs[i, j] = len(self.atom_list)
            _, pi = xs.sort(dim=1)
            for i, p in enumerate(pi):
                xx[i, :] = xx[i, p]
                aa[i, :, :] = aa[i, p, :]
                aa[i, :, :] = aa[i, :, p]
        z = flatten_graph(xx, aa)
        return self.network(z.to(self.device))


class GraphSPNZerokAry(nn.Module):
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

        self.atom_list = atom_list

        self.device = device
        self.to(device)

    def forward(self, x):
        xx = x['x']
        aa = x['a']
        n = math.comb(self.nd_nodes, self.arity)
        l = torch.zeros(len(xx), n, device=self.device)
        for i, pi in enumerate(itertools.combinations(range(self.nd_nodes), self.arity)):
            with torch.no_grad():
                px, pa = permute_graph(xx, aa, pi)
            z = flatten_graph(px, pa)
            l[:, i] = self.network(z.to(self.device)).squeeze()

        return torch.logsumexp(l, dim=1) - torch.log(torch.tensor(n))

    def logpdf(self, x):
        return self(x).mean()

    def sample(self, num_samples):
        num_sub_graphs = 20
        x = torch.randint(0, self.nk_nodes, (num_samples, self.nd_nodes),                dtype=torch.int)
        a = torch.randint(0, self.nk_edges, (num_samples, self.nd_nodes, self.nd_nodes), dtype=torch.int)

        n = math.comb(self.nd_nodes, self.arity)
        sub_graphs = list(itertools.combinations(range(self.nd_nodes), self.arity))
        sub_graphs = [sub_graphs[i] for i in torch.randint(n, (num_sub_graphs,)).tolist()]

        z = self.network.sample(num_sub_graphs*num_samples).to(torch.int).cpu()
        z = z.view(num_sub_graphs, num_samples, self.arity, self.arity+1)

        for g in range(num_sub_graphs):
            for i in range(self.arity):
                x[:, sub_graphs[g][i]] = z[g, :, i, 0]
                for j in range(self.arity):
                    a[:, sub_graphs[g][i], sub_graphs[g][j]] = z[g, :, i, j+1]

        return create_mols(x, a, self.atom_list)


class GraphSPNZeroFree(nn.Module):
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

        self.atom_list = atom_list

        self.device = device
        self.to(device)

    def forward(self, x):
        n = len(x['x'])
        xx = x['x'].view(n*self.nd_nodes)
        aa = x['a'].view(n*self.nd_nodes, self.nd_nodes)

        z = torch.cat((xx.unsqueeze(1), aa), dim=1)

        return self.network(z.to(self.device))

    def logpdf(self, x):
        return self(x).mean()

    def sample(self, num_samples):
        z = self.network.sample(num_samples*self.nd_nodes).to(torch.int).cpu()

        x = z[:, 0 ].view(-1, self.nd_nodes)
        a = z[:, 1:].view(-1, self.nd_nodes, self.nd_nodes)

        return create_mols(x, a, self.atom_list)

MODELS = {
    'graphspn_zero_none': GraphSPNZeroNone,
    'graphspn_zero_full': GraphSPNZeroFull,
    'graphspn_zero_rand': GraphSPNZeroRand,
    'graphspn_zero_sort': GraphSPNZeroSort,
    'graphspn_zero_kary': GraphSPNZerokAry,
    'graphspn_zero_free': GraphSPNZeroFree,
}
