import os
import torch
import torch.nn as nn

from abc import abstractmethod
from einsum import Graph, EinsumNetwork, ExponentialFamilyArray
from preprocess import MolecularDataset, load_qm9
from torch.distributions import Categorical
from rdkit import Chem
from rdkit.Chem.Draw import MolsToGridImage
from utils import *


def create_mols(x, a, atom_list):
    nd_nodes = x.size(1)
    mols = []
    smiles = []
    for x, a in zip(x, a):
        rw_mol = Chem.RWMol()

        for i in range(nd_nodes):
            if x[i].item() != 4:
                rw_mol.AddAtom(Chem.Atom(atom_decoder(atom_list)[x[i].item()]))

        num_atoms = rw_mol.GetNumAtoms()

        for i in range(num_atoms):
            for j in range(num_atoms):
                if a[i, j].item() != 3 and i > j:
                    rw_mol.AddBond(i, j, bond_decoder[a[i, j].item()])

                    flag, valence = valency(rw_mol)
                    if flag:
                        continue
                    else:
                        assert len(valence) == 2
                        k = valence[0]
                        v = valence[1]
                        atomic_number = rw_mol.GetAtomWithIdx(k).GetAtomicNum()
                        if atomic_number in (7, 8, 16) and (v - VALENCY_LIST[atomic_number]) == 1:
                            rw_mol.GetAtomWithIdx(k).SetFormalCharge(1)

        # rw_mol = radical_electrons_to_hydrogens(rw_mol)

        mols.append(rw_mol)
        smiles.append(Chem.MolToSmiles(rw_mol))

    return mols, smiles


class GraphSPNNaiveCore(nn.Module):
    def __init__(
        self, nc, nd_n, nd_e, nk_n, nk_e, ns_n, ns_e, ni_n, ni_e, graph_nodes, graph_edges, device, atom_list, regime
    ):
        super().__init__()
        self.nd_nodes = nd_n
        self.nd_edges = nd_e
        self.nk_nodes = nk_n
        self.nk_edges = nk_e
        self.regime = regime

        if regime == 'cat':
            ef_dist_n = ExponentialFamilyArray.CategoricalArray
            ef_dist_e = ExponentialFamilyArray.CategoricalArray
            ef_args_n = {'K': nk_n}
            ef_args_e = {'K': nk_e}
            num_dim_n = 1
            num_dim_e = 1
        elif regime == 'deq':
            ef_dist_n = ExponentialFamilyArray.NormalArray
            ef_dist_e = ExponentialFamilyArray.NormalArray
            ef_args_n = {'min_var': 1e-6, 'max_var': 0.01}
            ef_args_e = {'min_var': 1e-6, 'max_var': 0.01}
            num_dim_n = nk_n
            num_dim_e = nk_e
        else:
            os.error('Unsupported \'regime\'.')

        args_nodes = EinsumNetwork.Args(
            num_var=nd_n,
            num_dims=num_dim_n,
            num_input_distributions=ni_n,
            num_sums=ns_n,
            num_classes=nc,
            exponential_family=ef_dist_n,
            exponential_family_args=ef_args_n,
            use_em=False)
        args_edges = EinsumNetwork.Args(
            num_var=nd_e,
            num_dims=num_dim_e,
            num_input_distributions=ni_e,
            num_sums=ns_e,
            num_classes=nc,
            exponential_family=ef_dist_e,
            exponential_family_args=ef_args_e,
            use_em=False)

        self.network_nodes = EinsumNetwork.EinsumNetwork(graph_nodes, args_nodes)
        self.network_edges = EinsumNetwork.EinsumNetwork(graph_edges, args_edges)
        self.network_nodes.initialize()
        self.network_edges.initialize()

        self.atom_list = atom_list

        self.device = device
        self.to(device)

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def logpdf(self, x):
        pass

    @abstractmethod
    def sample(self, num_samples):
        pass


class GraphSPNNaiveCatA(GraphSPNNaiveCore):
    def __init__(
        self, nd_n, nd_e, nk_n, nk_e, nl_n, nl_e, nr_n, nr_e, ns_n, ns_e, ni_n, ni_e, device='cuda', atom_list=[6, 7, 8, 9]
    ):
        graph_nodes = Graph.random_binary_trees(nd_n, nl_n, nr_n)
        graph_edges = Graph.random_binary_trees(nd_e, nl_e, nr_e)

        super().__init__(1, nd_n, nd_e, nk_n, nk_e, ns_n, ns_e, ni_n, ni_e, graph_nodes, graph_edges, device, atom_list, 'cat')

    def forward(self, x):
        ll_nodes = self.network_nodes(x['x'].to(self.device))
        ll_edges = self.network_edges(x['a'].view(-1, self.nd_edges).to(self.device))
        return ll_nodes + ll_edges

    def logpdf(self, x):
        return self(x).mean()

    def sample(self, num_samples):
        x = self.network_nodes.sample(num_samples).cpu()
        a = self.network_edges.sample(num_samples).view(-1, self.nd_nodes, self.nd_nodes).cpu()
        x = x.to(torch.int)
        a = a.to(torch.int)

        return create_mols(x, a, self.atom_list)


class GraphSPNNaiveDeqA(GraphSPNNaiveCore):
    def __init__(
        self, nd_n, nd_e, nk_n, nk_e, nl_n, nl_e, nr_n, nr_e, ns_n, ns_e, ni_n, ni_e, device='cuda', atom_list=[6, 7, 8, 9]
    ):
        graph_nodes = Graph.random_binary_trees(nd_n, nl_n, nr_n)
        graph_edges = Graph.random_binary_trees(nd_e, nl_e, nr_e)

        super().__init__(1, nd_n, nd_e, nk_n, nk_e, ns_n, ns_e, ni_n, ni_e, graph_nodes, graph_edges, device, atom_list, 'deq')

    def forward(self, x):
        ll_nodes = self.network_nodes(x['x_deq'].to(self.device))
        ll_edges = self.network_edges(x['a_deq'].view(-1, self.nd_edges, self.nk_edges).to(self.device))
        return ll_nodes + ll_edges

    def logpdf(self, x):
        return self(x).mean()

    def sample(self, num_samples):
        x = self.network_nodes.sample(num_samples).cpu()
        a = self.network_edges.sample(num_samples).view(-1, self.nd_nodes, self.nd_nodes, self.nk_edges).cpu()
        x = x.argmax(2)
        a = a.argmax(3)

        return create_mols(x, a, self.atom_list)


class GraphSPNNaiveCatB(GraphSPNNaiveCore):
    def __init__(
        self, nd_n, nd_e, nk_n, nk_e, nl_n, nr_n, ns_n, ns_e, ni_n, ni_e, num_pieces, device='cuda', atom_list = [6, 7, 8, 9]
    ):
        graph_nodes = Graph.random_binary_trees(nd_n, nl_n, nr_n)
        graph_edges = Graph.poon_domingos_structure(shape=[nd_n, nd_n], delta=[[nd_n / d, nd_n / d] for d in num_pieces])

        super().__init__(1, nd_n, nd_e, nk_n, nk_e, ns_n, ns_e, ni_n, ni_e, graph_nodes, graph_edges, device, atom_list, 'cat')

    def forward(self, x):
        ll_nodes = self.network_nodes(x['x'].to(self.device))
        ll_edges = self.network_edges(x['a'].view(-1, self.nd_edges).to(self.device))
        return ll_nodes + ll_edges

    def logpdf(self, x):
        return self(x).mean()

    def sample(self, num_samples):
        x = self.network_nodes.sample(num_samples).cpu()
        a = self.network_edges.sample(num_samples).view(-1, self.nd_nodes, self.nd_nodes).cpu()

        x = x.to(torch.int)
        a = a.to(torch.int)

        return create_mols(x, a, self.atom_list)


class GraphSPNNaiveDeqB(GraphSPNNaiveCore):
    def __init__(
        self, nd_n, nd_e, nk_n, nk_e, nl_n, nr_n, ns_n, ns_e, ni_n, ni_e, num_pieces, device='cuda', atom_list = [6, 7, 8, 9]
    ):
        graph_nodes = Graph.random_binary_trees(nd_n, nl_n, nr_n)
        graph_edges = Graph.poon_domingos_structure(shape=[nd_n, nd_n], delta=[[nd_n / d, nd_n / d] for d in num_pieces])

        super().__init__(1, nd_n, nd_e, nk_n, nk_e, ns_n, ns_e, ni_n, ni_e, graph_nodes, graph_edges, device, atom_list, 'deq')

    def forward(self, x):
        ll_nodes = self.network_nodes(x['x_deq'].to(self.device))
        ll_edges = self.network_edges(x['a_deq'].view(-1, self.nd_edges, self.nk_edges).to(self.device))
        return ll_nodes + ll_edges

    def logpdf(self, x):
        return self(x).mean()

    def sample(self, num_samples):
        x = self.network_nodes.sample(num_samples).cpu()
        a = self.network_edges.sample(num_samples).view(-1, self.nd_nodes, self.nd_nodes, self.nk_edges).cpu()
        x = x.argmax(2)
        a = a.argmax(3)

        return create_mols(x, a, self.atom_list)


class GraphSPNNaiveCatC(GraphSPNNaiveCore):
    def __init__(
        self, nd_n, nk_n, nk_e, nl_n, nl_e, nr_n, nr_e, ns_n, ns_e, ni_n, ni_e, device='cuda', atom_list=[6, 7, 8, 9]
    ):
        nd_e = nd_n * (nd_n - 1) // 2

        graph_nodes = Graph.random_binary_trees(nd_n, nl_n, nr_n)
        graph_edges = Graph.random_binary_trees(nd_e, nl_e, nr_e)

        super().__init__(1, nd_n, nd_e, nk_n, nk_e, ns_n, ns_e, ni_n, ni_e, graph_nodes, graph_edges, device, atom_list, 'cat')

    def forward(self, x):
        a = x['a'].to(self.device)
        m = torch.tril(torch.ones_like(a, dtype=torch.bool), diagonal=-1)
        l = a[m].view(-1, self.nd_edges)

        ll_nodes = self.network_nodes(x['x'].to(self.device))
        ll_edges = self.network_edges(l)

        return ll_nodes + ll_edges

    def logpdf(self, x):
        return self(x).mean()

    def sample(self, num_samples):
        x = self.network_nodes.sample(num_samples).cpu()
        l = self.network_edges.sample(num_samples).cpu()

        x = x.to(torch.int)
        l = l.to(torch.int)

        m = torch.tril(torch.ones(num_samples, self.nd_nodes, self.nd_nodes, dtype=torch.bool), diagonal=-1)
        a = torch.zeros(num_samples, self.nd_nodes, self.nd_nodes, dtype=torch.int)
        a[m] = l.view(num_samples*self.nd_edges)

        return create_mols(x, a, self.atom_list)


class GraphSPNNaiveDeqC(GraphSPNNaiveCore):
    def __init__(
        self, nd_n, nk_n, nk_e, nl_n, nl_e, nr_n, nr_e, ns_n, ns_e, ni_n, ni_e, device='cuda', atom_list=[6, 7, 8, 9]
    ):
        nd_e = nd_n * (nd_n - 1) // 2

        graph_nodes = Graph.random_binary_trees(nd_n, nl_n, nr_n)
        graph_edges = Graph.random_binary_trees(nd_e, nl_e, nr_e)

        super().__init__(1, nd_n, nd_e, nk_n, nk_e, ns_n, ns_e, ni_n, ni_e, graph_nodes, graph_edges, device, atom_list, 'deq')

    def forward(self, x):
        a = x['a_deq'].to(self.device)
        m = torch.tril(torch.ones(self.nd_nodes, self.nd_nodes, dtype=torch.bool), diagonal=-1).unsqueeze(0).unsqueeze(3).expand_as(a)
        l = a[m].view(-1, self.nd_edges, self.nk_edges)

        ll_nodes = self.network_nodes(x['x_deq'].to(self.device))
        ll_edges = self.network_edges(l)

        return ll_nodes + ll_edges

    def logpdf(self, x):
        return self(x).mean()

    def sample(self, num_samples):
        x = self.network_nodes.sample(num_samples).cpu()
        l = self.network_edges.sample(num_samples).cpu()

        a = torch.zeros(num_samples, self.nd_nodes, self.nd_nodes, self.nk_edges)
        m = torch.tril(torch.ones(self.nd_nodes, self.nd_nodes, dtype=torch.bool), diagonal=-1).unsqueeze(0).unsqueeze(3).expand_as(a)
        a[m] = l.view(num_samples*self.nd_edges*self.nk_edges)

        x = x.argmax(2)
        a = a.argmax(3)

        return create_mols(x, a, self.atom_list)


class GraphSPNNaiveCatD(GraphSPNNaiveCore):
    def __init__(
        self, nc, nd_n, nk_n, nk_e, nl_n, nl_e, nr_n, nr_e, ns_n, ns_e, ni_n, ni_e, device='cuda', atom_list=[6, 7, 8, 9]
    ):
        nd_e = nd_n * (nd_n - 1) // 2

        graph_nodes = Graph.random_binary_trees(nd_n, nl_n, nr_n)
        graph_edges = Graph.random_binary_trees(nd_e, nl_e, nr_e)

        super().__init__(nc, nd_n, nd_e, nk_n, nk_e, ns_n, ns_e, ni_n, ni_e, graph_nodes, graph_edges, device, atom_list, 'cat')

        self.weights = nn.Parameter(torch.log_softmax(torch.randn(1, nc, device=self.device), dim=1), requires_grad=True)

    def forward(self, x):
        a = x['a'].to(self.device)
        m = torch.tril(torch.ones_like(a, dtype=torch.bool), diagonal=-1)
        l = a[m].view(-1, self.nd_edges)

        ll_nodes = self.network_nodes(x['x'].to(self.device))
        ll_edges = self.network_edges(l)

        return torch.logsumexp(ll_nodes + ll_edges + torch.log_softmax(self.weights, dim=1), dim=1)

    def logpdf(self, x):
        return self(x).mean()

    def sample(self, num_samples):
        x = torch.zeros(num_samples, self.nd_nodes)
        l = torch.zeros(num_samples, self.nd_edges)

        cs = Categorical(logits=self.weights).sample((num_samples, ))
        for i, c in enumerate(cs):
            x[i, :] = self.network_nodes.sample(1, class_idx=c).cpu()
            l[i, :] = self.network_edges.sample(1, class_idx=c).cpu()

        x = x.to(torch.int)
        l = l.to(torch.int)

        m = torch.tril(torch.ones(num_samples, self.nd_nodes, self.nd_nodes, dtype=torch.bool), diagonal=-1)
        a = torch.zeros(num_samples, self.nd_nodes, self.nd_nodes, dtype=torch.int)
        a[m] = l.view(num_samples*self.nd_edges)

        return create_mols(x, a, self.atom_list)


class GraphSPNNaiveDeqD(GraphSPNNaiveCore):
    def __init__(
        self, nc, nd_n, nk_n, nk_e, nl_n, nl_e, nr_n, nr_e, ns_n, ns_e, ni_n, ni_e, device='cuda', atom_list=[6, 7, 8, 9]
    ):
        nd_e = nd_n * (nd_n - 1) // 2

        graph_nodes = Graph.random_binary_trees(nd_n, nl_n, nr_n)
        graph_edges = Graph.random_binary_trees(nd_e, nl_e, nr_e)

        super().__init__(nc, nd_n, nd_e, nk_n, nk_e, ns_n, ns_e, ni_n, ni_e, graph_nodes, graph_edges, device, atom_list, 'deq')

        self.weights = nn.Parameter(torch.log_softmax(torch.randn(1, nc, device=self.device), dim=1), requires_grad=True)

    def forward(self, x):
        a = x['a_deq'].to(self.device)
        m = torch.tril(torch.ones_like(a, dtype=torch.bool), diagonal=-1)
        l = a[m].view(-1, self.nd_edges, self.nk_edges)

        ll_nodes = self.network_nodes(x['x_deq'].to(self.device))
        ll_edges = self.network_edges(l)

        return torch.logsumexp(ll_nodes + ll_edges + torch.log_softmax(self.weights, dim=1), dim=1)

    def logpdf(self, x):
        return self(x).mean()

    def sample(self, num_samples):
        x = torch.zeros(num_samples, self.nd_nodes, self.nk_nodes)
        l = torch.zeros(num_samples, self.nd_edges, self.nk_edges)

        cs = Categorical(logits=self.weights).sample((num_samples, ))
        for i, c in enumerate(cs):
            x[i, :] = self.network_nodes.sample(1, class_idx=c).cpu()
            l[i, :] = self.network_edges.sample(1, class_idx=c).cpu()

        m = torch.tril(torch.ones(num_samples, self.nd_nodes, self.nd_nodes, self.nk_edges, dtype=torch.bool), diagonal=-1)
        a = torch.zeros(num_samples, self.nd_nodes, self.nd_nodes, self.nk_edges, dtype=torch.int)
        a[m] = l.view(num_samples*self.nd_edges*self.nk_edges)

        x = x.argmax(2)
        a = a.argmax(3)

        return create_mols(x, a, self.atom_list)


class GraphSPNNaiveCatE(nn.Module):
    def __init__(
        self, nd_n, nd_e, nk_n, nk_e, ns, ni, nl, nr, device='cuda', atom_list=[6, 7, 8, 9]
    ):
        super().__init__()
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
        z = torch.cat((x['x'].unsqueeze(1).to(self.device), x['a'].to(self.device)), dim=1)
        z = z.view(-1, self.nd_nodes + self.nd_edges).to(self.device)
        return self.network(z)

    def logpdf(self, x):
        return self(x).mean()

    def sample(self, num_samples):
        z = self.network.sample(num_samples).to(torch.int).cpu()

        z = z.view(-1, self.nd_nodes+1, self.nd_nodes)
        x = z[:, 0 , :]
        a = z[:, 0:, :]

        return create_mols(x, a, self.atom_list)


class GraphSPNNaiveCatF(nn.Module):
    def __init__(
        self, nd_n, nd_e, nk_n, nk_e, ns, ni, num_pieces, device='cuda', atom_list=[6, 7, 8, 9]
    ):
        super().__init__()
        self.nd_nodes = nd_n
        self.nd_edges = nd_e

        nd = nd_n + nd_e
        nk = max(nk_n, nk_e)

        graph = Graph.poon_domingos_structure(shape=[nd_n+1, nd_n], delta=[[(nd_n+1) / d, nd_n / d] for d in num_pieces])

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
        z = torch.cat((x['x'].unsqueeze(1).to(self.device), x['a'].to(self.device)), dim=1)
        z = z.view(-1, self.nd_nodes + self.nd_edges).to(self.device)
        return self.network(z)

    def logpdf(self, x):
        return self(x).mean()

    def sample(self, num_samples):
        z = self.network.sample(num_samples).to(torch.int).cpu()

        z = z.view(-1, self.nd_nodes+1, self.nd_nodes)
        x = z[:, 0 , :]
        a = z[:, 0:, :]

        return create_mols(x, a, self.atom_list)


class GraphSPNNaiveCatG(nn.Module):
    def __init__(
        self, nd_n, nk_n, nk_e, ns, ni, nl, nr, device='cuda', atom_list=[6, 7, 8, 9]
    ):
        super().__init__()
        nd_e = nd_n * (nd_n - 1) // 2

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
        a = x['a'].to(self.device)
        m = torch.tril(torch.ones_like(a, dtype=torch.bool), diagonal=-1)
        l = a[m].view(-1, self.nd_edges)

        z = torch.cat((x['x'].to(self.device), l.to(self.device)), dim=1)
        return self.network(z)

    def logpdf(self, x):
        return self(x).mean()

    def sample(self, num_samples):
        z = self.network.sample(num_samples).to(torch.int).cpu()

        x = z[:, :self.nd_nodes]
        l = z[:, self.nd_nodes:]

        m = torch.tril(torch.ones(num_samples, self.nd_nodes, self.nd_nodes, dtype=torch.bool), diagonal=-1)
        a = torch.zeros(num_samples, self.nd_nodes, self.nd_nodes, dtype=torch.int)
        a[m] = l.reshape(num_samples*self.nd_edges)

        return create_mols(x, a, self.atom_list)


class GraphSPNNaiveCatH(nn.Module):
    def __init__(
        self, nc, nd_n, nk_n, nk_e, nl_n, nl_e, nr_n, nr_e, ns_n, ns_e, ni_n, ni_e, device='cuda', atom_list=[6, 7, 8, 9]
    ):
        super().__init__()
        nd_e = nd_n * (nd_n - 1) // 2

        self.nc = nc

        self.nd_nodes = nd_n
        self.nd_edges = nd_e

        self.network_nodes = nn.ParameterList([])
        self.network_edges = nn.ParameterList([])

        for _ in range(nc):
            graph_nodes = Graph.random_binary_trees(nd_n, nl_n, nr_n)
            graph_edges = Graph.random_binary_trees(nd_e, nl_e, nr_e)

            args_nodes = EinsumNetwork.Args(
                num_var=nd_n,
                num_input_distributions=ni_n,
                num_sums=ns_n,
                exponential_family=ExponentialFamilyArray.CategoricalArray,
                exponential_family_args={'K': nk_n},
                use_em=False)
            args_edges = EinsumNetwork.Args(
                num_var=nd_e,
                num_input_distributions=ni_e,
                num_sums=ns_e,
                exponential_family=ExponentialFamilyArray.CategoricalArray,
                exponential_family_args={'K': nk_e},
                use_em=False)

            net_n = EinsumNetwork.EinsumNetwork(graph_nodes, args_nodes)
            net_e = EinsumNetwork.EinsumNetwork(graph_edges, args_edges)
            net_n.initialize()
            net_e.initialize()
            net_n.to(device)
            net_e.to(device)

            self.network_nodes.append(net_n)
            self.network_edges.append(net_e)

        self.weights = nn.Parameter(torch.log_softmax(torch.randn(1, nc, device=device), dim=1), requires_grad=True)

        self.atom_list = atom_list

        self.device = device
        self.to(device)

    def forward(self, x):
        a = x['a'].to(self.device)
        m = torch.tril(torch.ones_like(a, dtype=torch.bool), diagonal=-1)
        l = a[m].view(-1, self.nd_edges)

        z = x['x'].to(self.device)
        batch_size = len(z)

        # ll_nodes = torch.cat([net(z) for net in self.network_nodes], dim=1)
        # ll_edges = torch.cat([net(l) for net in self.network_edges], dim=1)
        ll_nodes = torch.zeros(batch_size, self.nc, device=self.device)
        ll_edges = torch.zeros(batch_size, self.nc, device=self.device)
        for c, net in enumerate(self.network_nodes):
            ll_nodes[:, c] = net(z).squeeze()
        for c, net in enumerate(self.network_edges):
            ll_edges[:, c] = net(l).squeeze()

        return torch.logsumexp(ll_nodes + ll_edges + torch.log_softmax(self.weights, dim=1), dim=1)

    def logpdf(self, x):
        return self(x).mean()

    def sample(self, num_samples):
        # c = Categorical(logits=self.weights).sample((1, ))
        # x = self.network_nodes.sample(num_samples, class_idx=c).cpu()
        # l = self.network_edges.sample(num_samples, class_idx=c).cpu()

        x = torch.zeros(num_samples, self.nd_nodes)
        l = torch.zeros(num_samples, self.nd_edges)

        cs = Categorical(logits=self.weights).sample((num_samples, ))
        for i, c in enumerate(cs):
            x[i, :] = self.network_nodes[c].sample(1).cpu()
            l[i, :] = self.network_edges[c].sample(1).cpu()

        x = x.to(torch.int)
        l = l.to(torch.int)

        m = torch.tril(torch.ones(num_samples, self.nd_nodes, self.nd_nodes, dtype=torch.bool), diagonal=-1)
        a = torch.zeros(num_samples, self.nd_nodes, self.nd_nodes, dtype=torch.int)
        a[m] = l.view(num_samples*self.nd_edges)

        return create_mols(x, a, self.atom_list)


MODELS = {
    'graphspn_naive_cat_a': GraphSPNNaiveCatA,
    'graphspn_naive_cat_b': GraphSPNNaiveCatB,
    'graphspn_naive_cat_c': GraphSPNNaiveCatC,
    'graphspn_naive_cat_d': GraphSPNNaiveCatD,
    'graphspn_naive_cat_e': GraphSPNNaiveCatE,
    'graphspn_naive_cat_f': GraphSPNNaiveCatF,
    'graphspn_naive_cat_g': GraphSPNNaiveCatG,
    'graphspn_naive_cat_h': GraphSPNNaiveCatH,
    'graphspn_naive_deq_a': GraphSPNNaiveDeqA,
    'graphspn_naive_deq_b': GraphSPNNaiveDeqB,
    'graphspn_naive_deq_c': GraphSPNNaiveDeqC,
    'graphspn_naive_deq_d': GraphSPNNaiveDeqD,
}


if __name__ == '__main__':
    checkpoint_dir = 'results/training/model_checkpoint/'
    evaluation_dir = 'results/training/model_evaluation/'

    name = 'graphspn_naive_deq_c'

    x_trn, _, _ = load_qm9(0, raw=True)
    smiles_trn = [x['s'] for x in x_trn]

    model_path = best_model(evaluation_dir + name + '/')[0]
    model_best = torch.load(checkpoint_dir + name + '/' + model_path)

    molecules_gen, smiles_gen = model_best.sample(1000)

    results = evaluate(molecules_gen, smiles_gen, smiles_trn, 1000, return_unique=True, debug=False)

    img = MolsToGridImage(mols=results['mols_valid'][0:100], molsPerRow=10, subImgSize=(200, 200), useSVG=False)
    img.save(f'sampling.png')
