import math
import torch
import torch.nn as nn
import itertools

from tqdm import tqdm
from abc import abstractmethod
from einsum import Graph, EinsumNetwork, ExponentialFamilyArray
from utils.graphs import flatten_graph, unflatt_graph, permute_graph
from models.spn_utils import ohe2cat, cat2ohe


def permute_graph_per_instance(x, a, nd_nodes, nk_nodes):
    px = torch.zeros_like(x)
    pa = torch.zeros_like(a)
    for i in range(len(x)):
        num_full = torch.sum(x[i, :] != nk_nodes-1)
        pi = torch.cat((torch.randperm(num_full), torch.arange(num_full, nd_nodes)))
        px[i, :] = x[i, pi]
        pa[i, :, :] = a[i, pi, :]
        pa[i, :, :] = a[i, :, pi]
    return px, pa

def permute_graph_per_batch(x, a, nd_nodes, nk_nodes, permutation):
    px = torch.zeros_like(x)
    pa = torch.zeros_like(a)
    permutation = torch.tensor(permutation)
    for i in range(len(x)):
        num_full = torch.sum(x[i, :] != nk_nodes-1)
        pi = torch.cat((permutation[permutation < num_full.cpu()], torch.arange(num_full, nd_nodes)))
        px[i, :] = x[i, pi]
        pa[i, :, :] = a[i, pi, :]
        pa[i, :, :] = a[i, :, pi]
    return px, pa


class GraphSPNZeroCore(nn.Module):
    def __init__(self, nd_n, nk_n, nk_e, ns, ni, nl, nr, device='cuda'):
        super().__init__()
        nd_e = nd_n**2
        self.nd_nodes = nd_n
        self.nk_nodes = nk_n
        self.nk_edges = nk_e

        nd = nd_n + nd_e
        # The implementation of the Einsum networks does not allow for hybrid
        # probability distributions in the input layer (e.g., two Categorical
        # distributions with different number of categories). Therefore, we have
        # to take the maximum number of categories and then truncate the adjacency
        # matrix when sampling the bonds (as also mentioned below).
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

        self.device = device
        self.to(device)

    @abstractmethod
    def _forward(self, x, a):
        pass

    def forward(self, x, a):
        x, a = ohe2cat(x, a)
        return self._forward(x, a)

    def logpdf(self, x, a):
        return self(x, a).mean()

    def sample(self, num_samples):
        z = self.network.sample(num_samples).to(torch.int).cpu()
        x, a = unflatt_graph(z, self.nd_nodes, self.nd_nodes)
        # We have to truncate the adjacency matrix since the implementation of the
        # Einsum networks does not allow for two different Categorical distributions
        # in the input layer (as explained above).
        a[a > 3] = 3
        return cat2ohe(x, a, self.nk_nodes, self.nk_edges)


class GraphSPNZeroNone(GraphSPNZeroCore):
    def __init__(self, nd_n, nk_n, nk_e, ns, ni, nl, nr, device='cuda'):
        super().__init__(nd_n, nk_n, nk_e, ns, ni, nl, nr, device)

    def _forward(self, x, a):
        return self.network(flatten_graph(x, a).to(self.device))


class GraphSPNZeroFull(GraphSPNZeroCore):
    def __init__(self, nd_n, nk_n, nk_e, ns, ni, nl, nr, device='cuda'):
        super().__init__(nd_n, nk_n, nk_e, ns, ni, nl, nr, device)

    def _forward(self, x, a):
        n = torch.tensor(math.factorial(self.nd_nodes))
        l = torch.zeros(len(x), device=self.device)
        for pi in tqdm(itertools.permutations(range(self.nd_nodes), self.nd_nodes), leave=False):
            with torch.no_grad():
                px, pa = permute_graph_per_batch(x, a, self.nd_nodes, self.nk_nodes, pi)
            l += (torch.exp(self.network(flatten_graph(px, pa).to(self.device)).squeeze() - torch.log(n)))
        return torch.log(l)


class GraphSPNZeroRand(GraphSPNZeroCore):
    def __init__(self, nd_n, nk_n, nk_e, ns, ni, nl, nr, np, device='cuda'):
        super().__init__(nd_n, nk_n, nk_e, ns, ni, nl, nr, device)

        self.num_perms = min(np, math.factorial(nd_n))
        self.permutations = torch.stack([torch.randperm(nd_n) for _ in range(self.num_perms)])
        self.i_perm = 0

    def _forward(self, x, a):
        # l = torch.zeros(len(x), self.num_perms, device=self.device)
        # for i, pi in enumerate(torch.stack([torch.randperm(self.nd_nodes) for _ in range(self.num_perms)])):
        #     with torch.no_grad():
        #         px, pa = permute_graph_per_batch(x, a, self.nd_nodes, self.nk_nodes, pi)
        #     l[:, i] = self.network(flatten_graph(px, pa).to(self.device)).squeeze()
        with torch.no_grad():
            px, pa = permute_graph_per_batch(x, a, self.nd_nodes, self.nk_nodes, self.permutations[self.i_perm])
            self.i_perm += 1
            if self.i_perm == self.num_perms:
                self.i_perm = 0
        return self.network(flatten_graph(px, pa).to(self.device)).squeeze()


class GraphSPNZeroSort(GraphSPNZeroCore):
    def __init__(self, nd_n, nk_n, nk_e, ns, ni, nl, nr, device='cuda'):
        super().__init__(nd_n, nk_n, nk_e, ns, ni, nl, nr, device)

    def _forward(self, x, a):
        # with torch.no_grad():
            # xx, pi = xx.sort(dim=1)
            # for i, p in enumerate(pi):
            #     aa[i, :, :] = aa[i, p, :]
            #     aa[i, :, :] = aa[i, :, p]

            # xs = torch.zeros_like(xx)
            # for i in range(len(xx)):
            #     num_full = torch.sum(xx[i, :] != len(self.atom_list))
            #     for j in range(self.nd_nodes):
            #         if (j < num_full) and (xx[i, j] < len(self.atom_list)):
            #             xs[i, j] = self.atom_list[xx[i, j]]
            #         else:
            #             xs[i, j] = len(self.atom_list)
            # _, pi = xs.sort(dim=1)
            # for i, p in enumerate(pi):
            #     xx[i, :] = xx[i, p]
            #     aa[i, :, :] = aa[i, p, :]
            #     aa[i, :, :] = aa[i, :, p]

            # Impose the canonical ordering
            # xx, aa = create_graphs(create_mols(xx, aa, self.atom_list, canonical=True)[0], self.nd_nodes, self.atom_list)

            # for i, mol in enumerate(create_mols(xx, aa, self.atom_list)[0]):
            #     p = list(Chem.CanonicalRankAtoms(mol))
            #     p = p + list(arange(len(p), self.nd_nodes))
            #     xx[i, :] = xx[i, p]
            #     aa[i, :, :] = aa[i, p, :]
            #     aa[i, :, :] = aa[i, :, p]

            # It is better to impose the canonical ordering before the training.

        return self.network(flatten_graph(x, a).to(self.device))


class GraphSPNZerokAry(nn.Module):
    def __init__(self, nd_n, nk_n, nk_e, ns, ni, nl, nr, arity, device='cuda'):
        super().__init__()
        self.nd_nodes = nd_n
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

        self.device = device
        self.to(device)

    def forward(self, x, a):
        x, a = ohe2cat(x, a)
        num_perms = math.perm(self.nd_nodes, self.arity)
        l = torch.zeros(len(x), num_perms, device=self.device)
        for i, pi in enumerate(itertools.permutations(range(self.nd_nodes), self.arity)):
            with torch.no_grad():
                px, pa = permute_graph(x, a, pi)
            l[:, i] = self.network(flatten_graph(px, pa).to(self.device)).squeeze()
        return torch.logsumexp(l, dim=1) - torch.log(torch.tensor(num_perms))

    def logpdf(self, x, a):
        return self(x, a).mean()

    def sample(self, num_samples):
        num_sub_graphs = 9
        x = torch.randint(0, self.nk_nodes, (num_samples, self.nd_nodes),                dtype=torch.int)
        a = torch.randint(0, self.nk_edges, (num_samples, self.nd_nodes, self.nd_nodes), dtype=torch.int)

        num_perms = math.perm(self.nd_nodes, self.arity)
        sub_graphs = list(itertools.permutations(range(self.nd_nodes), self.arity))
        sub_graphs = [sub_graphs[i] for i in torch.randint(num_perms, (num_sub_graphs,)).tolist()]

        z = self.network.sample(num_sub_graphs*num_samples).to(torch.int).cpu()
        z = z.view(num_sub_graphs, num_samples, self.arity, self.arity+1)

        for g in range(num_sub_graphs):
            for i in range(self.arity):
                x[:, sub_graphs[g][i]] = z[g, :, i, 0]
                for j in range(self.arity):
                    a[:, sub_graphs[g][i], sub_graphs[g][j]] = z[g, :, i, j+1]

        return cat2ohe(x, a, self.nk_nodes, self.nk_edges)


class GraphSPNZeroFree(nn.Module):
    def __init__(self, nd_n, nk_n, nk_e, ns, ni, nl, nr, device='cuda'):
        super().__init__()
        self.nd_nodes = nd_n
        self.nk_nodes = nk_n
        self.nk_edges = nk_e

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

        self.device = device
        self.to(device)

    def forward(self, x, a):
        n = len(x)
        x, a = ohe2cat(x, a)

        xx = x.view(n*self.nd_nodes)
        aa = a.view(n*self.nd_nodes, self.nd_nodes)

        z = torch.cat((xx.unsqueeze(1), aa), dim=1)

        return self.network(z.to(self.device))

    def logpdf(self, x, a):
        return self(x, a).mean()

    def sample(self, num_samples):
        z = self.network.sample(num_samples*self.nd_nodes).to(torch.int).cpu()

        x = z[:, 0 ].view(-1, self.nd_nodes)
        a = z[:, 1:].view(-1, self.nd_nodes, self.nd_nodes)
        a[a > 3] = 3
        return cat2ohe(x, a, self.nk_nodes, self.nk_edges)

MODELS = {
    'graphspn_zero_none': GraphSPNZeroNone,
    'graphspn_zero_full': GraphSPNZeroFull,
    'graphspn_zero_rand': GraphSPNZeroRand,
    'graphspn_zero_sort': GraphSPNZeroSort,
    'graphspn_zero_kary': GraphSPNZerokAry,
    'graphspn_zero_free': GraphSPNZeroFree,
}
