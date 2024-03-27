import torch
import torch.nn as nn

from einsum import Graph, EinsumNetwork, ExponentialFamilyArray
from preprocess import MolecularDataset, load_qm9
from rdkit import Chem
from rdkit.Chem.Draw import MolsToGridImage
from utils import *


class GraphSPN(nn.Module):
    def __init__(
        self,
        nd_n,
        nd_e,
        nk_n,
        nk_e,
        nl_n,
        nl_e,
        nr_n,
        nr_e,
        ns_n,
        ns_e,
        ni_n,
        ni_e,
        device='cuda',
        atom_list = [6, 7, 8, 9]
    ):
        super().__init__()
        self.nd_nodes = nd_n
        self.nd_edges = nd_e

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

        self.network_nodes = EinsumNetwork.EinsumNetwork(graph_nodes, args_nodes)
        self.network_edges = EinsumNetwork.EinsumNetwork(graph_edges, args_edges)
        self.network_nodes.initialize()
        self.network_edges.initialize()

        self.atom_list = atom_list

        self.device = device
        self.to(device)

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

        mols = []
        smiles = []
        for x, a in zip(x, a):
            rw_mol = Chem.RWMol()

            for i in range(self.nd_nodes):
                if x[i].item() != 4:
                    rw_mol.AddAtom(Chem.Atom(atom_decoder(self.atom_list)[x[i].item()]))

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


if __name__ == '__main__':
    x_trn, _, _ = load_qm9(0, raw=True)
    smiles_trn = [x['s'] for x in x_trn]

    model = torch.load('results/training/model_checkpoint/graphspn_naive/dataset=qm9_property_model=graphspn_naive_nd_n=9_nd_e=81_nk_n=5_nk_e=4_nl_n=3_nl_e=3_nr_n=10_nr_e=10_ns_n=10_ns_e=10_ni_n=5_ni_e=5_device=cuda_optimizer=adam_lr=0.05_betas=[0.9, 0.82]_num_epochs=20_batch_size=100_seed=0.pt')

    molecules_gen, smiles_gen = model.sample(1000)

    results = evaluate(molecules_gen, smiles_gen, smiles_trn, 1000, return_unique=True, debug=False)

    img = MolsToGridImage(mols=results['mols_valid'][0:100], molsPerRow=10, subImgSize=(200, 200), useSVG=False)
    img.save(f'sampling.png')
