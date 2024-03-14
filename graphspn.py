import json
import torch
import torch.nn as nn

from einsum import Graph, EinsumNetwork, ExponentialFamilyArray
from preprocess import MolecularDataset, load_qm9

class GraphSPN(nn.Module):
    def __init__(
        self,
        nd_nodes,
        nd_edges,
        nk_nodes,
        nk_edges,
        nl_nodes,
        nl_edges,
        nr_nodes,
        nr_edges,
        ns_nodes,
        ns_edges,
        ni_nodes,
        ni_edges,
        device='cuda'
    ):
        super().__init__()
        self.nd_nodes = nd_nodes
        self.nd_edges = nd_edges

        graph_nodes = Graph.random_binary_trees(nd_nodes, nl_nodes, nr_nodes)
        graph_edges = Graph.random_binary_trees(nd_edges, nl_edges, nr_edges)

        args_nodes = EinsumNetwork.Args(
            num_var=nd_nodes,
            num_input_distributions=ni_nodes,
            num_sums=ns_nodes,
            exponential_family=ExponentialFamilyArray.CategoricalArray,
            exponential_family_args={'K': nk_nodes},
            use_em=False)
        args_edges = EinsumNetwork.Args(
            num_var=nd_edges,
            num_input_distributions=ni_edges,
            num_sums=ns_edges,
            exponential_family=ExponentialFamilyArray.CategoricalArray,
            exponential_family_args={'K': nk_edges},
            use_em=False)

        self.network_nodes = EinsumNetwork.EinsumNetwork(graph_nodes, args_nodes)
        self.network_edges = EinsumNetwork.EinsumNetwork(graph_edges, args_edges)
        self.network_nodes.initialize()
        self.network_edges.initialize()

        self.device = device
        self.to(device)

    def forward(self, x):
        ll_nodes = self.network_nodes(x['x'].to(self.device))
        ll_edges = self.network_edges(x['a'].view(-1, self.nd_edges).to(self.device))
        return ll_nodes + ll_edges

    def logpdf(self, x):
        return self(x).mean()

if __name__ == '__main__':
    loader_trn, loader_val, loader_tst = load_qm9(10)
    name = 'graphspn'

    with open('config/' + f'{name}.json', 'r') as fp:
        hyperpars = json.load(fp)

    model = GraphSPN(**hyperpars['model_hyperpars'])

    print(model)

    for batch in loader_trn:
        print(model(batch))