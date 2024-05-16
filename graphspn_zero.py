import torch
import torch.nn as nn

from einsum import Graph, EinsumNetwork, ExponentialFamilyArray
from utils import *

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
        _x = x['x'].view(n*self.nd_nodes)
        _a = x['a'].view(n*self.nd_nodes, self.nd_nodes)

        z = torch.cat((_x.unsqueeze(1), _a), dim=1)

        return self.network(z.to(self.device))

    def logpdf(self, x):
        return self(x).mean()

    def sample(self, num_samples):
        z = self.network.sample(num_samples*self.nd_nodes).to(torch.int).cpu()

        x = z[:, 0 ].view(-1, self.nd_nodes)
        a = z[:, 1:].view(-1, self.nd_nodes, self.nd_nodes)

        return create_mols(x, a, self.atom_list)

MODELS = {
    'graphspn_zero_free': GraphSPNZeroFree,
}
