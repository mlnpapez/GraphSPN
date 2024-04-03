from numpy import zeros
import torch
import torch.nn as nn

from abc import abstractmethod
from einsum import Graph, EinsumNetwork, ExponentialFamilyArray
from preprocess import MolecularDataset, load_qm9
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
        self, nd_n, nd_e, nk_n, nk_e, ns_n, ns_e, ni_n, ni_e, graph_nodes, graph_edges, device, atom_list
    ):
        super().__init__()
        self.nd_nodes = nd_n
        self.nd_edges = nd_e

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


class GraphSPNNaiveA(GraphSPNNaiveCore):
    def __init__(
        self, nd_n, nd_e, nk_n, nk_e, nl_n, nl_e, nr_n, nr_e, ns_n, ns_e, ni_n, ni_e, device='cuda', atom_list=[6, 7, 8, 9]
    ):
        graph_nodes = Graph.random_binary_trees(nd_n, nl_n, nr_n)
        graph_edges = Graph.random_binary_trees(nd_e, nl_e, nr_e)

        super().__init__(nd_n, nd_e, nk_n, nk_e, ns_n, ns_e, ni_n, ni_e, graph_nodes, graph_edges, device, atom_list)

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


class GraphSPNNaiveB(GraphSPNNaiveCore):
    def __init__(
        self, nd_n, nd_e, nk_n, nk_e, nl_n, nr_n, ns_n, ns_e, ni_n, ni_e, num_pieces, device='cuda', atom_list = [6, 7, 8, 9]
    ):
        graph_nodes = Graph.random_binary_trees(nd_n, nl_n, nr_n)
        graph_edges = Graph.poon_domingos_structure(shape=[nd_n, nd_n], delta=[[nd_n / d, nd_n / d] for d in num_pieces])

        super().__init__(nd_n, nd_e, nk_n, nk_e, ns_n, ns_e, ni_n, ni_e, graph_nodes, graph_edges, device, atom_list)

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


class GraphSPNNaiveC(GraphSPNNaiveCore):
    def __init__(
        self, nd_n, nk_n, nk_e, nl_n, nl_e, nr_n, nr_e, ns_n, ns_e, ni_n, ni_e, device='cuda', atom_list=[6, 7, 8, 9]
    ):
        nd_e = nd_n * (nd_n - 1) // 2

        graph_nodes = Graph.random_binary_trees(nd_n, nl_n, nr_n)
        graph_edges = Graph.random_binary_trees(nd_e, nl_e, nr_e)

        super().__init__(nd_n, nd_e, nk_n, nk_e, ns_n, ns_e, ni_n, ni_e, graph_nodes, graph_edges, device, atom_list)

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


class GraphSPNNaiveD(nn.Module):
    def __init__(
        self, nd_n, nd_e, nk_n, nk_e, ns, ni, nl, nr, device='cuda', atom_list=[6, 7, 8, 9]
    ):
        super().__init__()
        self.nd_nodes = nd_n
        self.nd_edges = nd_e
        self.nk_nodes = nk_n
        self.nk_edges = nk_e

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


class GraphSPNNaiveE(nn.Module):
    def __init__(
        self, nd_n, nd_e, nk_n, nk_e, ns, ni, num_pieces, device='cuda', atom_list=[6, 7, 8, 9]
    ):
        super().__init__()
        self.nd_nodes = nd_n
        self.nd_edges = nd_e
        self.nk_nodes = nk_n
        self.nk_edges = nk_e

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


class GraphSPNNaiveF(nn.Module):
    def __init__(
        self, nd_n, nk_n, nk_e, ns, ni, nl, nr, device='cuda', atom_list=[6, 7, 8, 9]
    ):
        super().__init__()
        nd_e = nd_n * (nd_n - 1) // 2

        self.nd_nodes = nd_n
        self.nd_edges = nd_e
        self.nk_nodes = nk_n
        self.nk_edges = nk_e

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


MODELS = {
    'graphspn_naive_a': GraphSPNNaiveA,
    'graphspn_naive_b': GraphSPNNaiveB,
    'graphspn_naive_c': GraphSPNNaiveC,
    'graphspn_naive_d': GraphSPNNaiveD,
    'graphspn_naive_d': GraphSPNNaiveE,
    'graphspn_naive_f': GraphSPNNaiveF
}


if __name__ == '__main__':
    checkpoint_dir = 'results/training/model_checkpoint/'
    evaluation_dir = 'results/training/model_evaluation/'

    name = 'graphspn_naive_f'

    x_trn, _, _ = load_qm9(0, raw=True)
    smiles_trn = [x['s'] for x in x_trn]

    model_path = best_model(evaluation_dir + name + '/')[0]
    model_best = torch.load(checkpoint_dir + name + '/' + model_path)

    molecules_gen, smiles_gen = model_best.sample(1000)

    results = evaluate(molecules_gen, smiles_gen, smiles_trn, 1000, return_unique=True, debug=False)

    img = MolsToGridImage(mols=results['mols_valid'][0:100], molsPerRow=10, subImgSize=(200, 200), useSVG=False)
    img.save(f'sampling.png')
