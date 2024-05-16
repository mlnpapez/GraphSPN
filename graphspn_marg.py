import torch
import torch.nn as nn

from einsum import Graph, EinsumNetwork, ExponentialFamilyArray
from utils import *

class GraphSPNMargNone(nn.Module):
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

    def forward(self, x):
        o = []
        c = torch.count_nonzero(x['x'] == len(self.atom_list), dim=1)
        for num_empty in torch.unique(c):
            num_full = self.nd_nodes-num_empty.item()
            xx = x['x'][c == num_empty]
            aa = x['a'][c == num_empty]
            with torch.no_grad():
                if num_empty > 0:
                    mx = torch.zeros(self.nd_nodes,                dtype=torch.bool)
                    ma = torch.zeros(self.nd_nodes, self.nd_nodes, dtype=torch.bool)
                    mx[num_full:   ] = True
                    ma[num_full:, :] = True
                    ma[:, num_full:] = True
                    m = torch.cat((mx.unsqueeze(1), ma), dim=1)
                    marginalization_idx = torch.arange(self.nd_nodes+self.nd_edges, requires_grad=False)[m.view(-1)]

                    self.network.set_marginalization_idx(marginalization_idx)
                else:
                    self.network.set_marginalization_idx(None)

            z = torch.cat((xx.unsqueeze(2), aa), dim=2)
            z = z.view(-1, self.nd_nodes + self.nd_edges).to(self.device)
            o.append(self.network(z))

        return torch.cat(o)

    def logpdf(self, x):
        return self(x).mean()

    def sample(self, num_samples):
        z = self.network.sample(num_samples).to(torch.int).cpu()

        z = z.view(-1, self.nd_nodes, self.nd_nodes+1)
        x = z[:, :, 0 ]
        a = z[:, :, 1:]

        return create_mols(x, a, self.atom_list)


MODELS = {
    'graphspn_marg_none': GraphSPNMargNone,
}
