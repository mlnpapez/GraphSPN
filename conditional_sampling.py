import torch
from utils.datasets import MOLECULAR_DATASETS
from utils.molecular import isvalid, mol2g, gs2mols
from utils.plot import grid_conditional, grid_unconditional
from utils.graphs import flatten_graph, unflatt_graph
from models.spn_utils import ohe2cat, cat2ohe


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


from rdkit import Chem, rdBase
rdBase.DisableLog("rdApp.error")

def conditional_sample(model, xx, aa, submol_size, num_samples, max_types, atom_list):
    """Conditionaly generate a molecule given some other (smaller) molecule.
    Parameters:
        model: GraphSPN model
        xx (torch.Tensor): feature tensor [1, max_size]
        aa (torch.Tensor): adjacency tensor [1, max_size^2]
        submol_size (int): number of atoms of molecule made from 'xx' and 'aa'
        num_samples (int): number of molecules to sample
    Returns:
        mol, sml (tuple[list, list]): conditionaly generated molecules\smiles
    """
    # NOTE: accepts only one observation as an input
    marginalize(model.network, model.nd_nodes, xx.shape[-1]-submol_size, submol_size)
    z = flatten_graph(xx, aa)

    z = z.to(model.device)
    with torch.no_grad():
        z = z.expand(num_samples, -1)
        sample = model.network.sample(x=z.to(torch.float)).cpu()
        xx_sample, aa_sample = unflatt_graph(sample, model.nd_nodes, model.nd_nodes)
        xx_sample, aa_sample = cat2ohe(xx_sample, aa_sample, max_types, 4)
        mol_sample = gs2mols(xx_sample, aa_sample, atom_list)
        sml_sample = [Chem.MolToSmiles(mol, kekuleSmiles=True) for mol in mol_sample]

    return mol_sample, sml_sample

def create_observed_mol(smile, max_atoms, atom_list):
    mol = Chem.MolFromSmiles(smile)
    Chem.Kekulize(mol)
    xx, aa = mol2g(mol, max_atoms, atom_list)
    return *ohe2cat(xx.unsqueeze(0), aa.unsqueeze(0)), mol.GetNumAtoms()

if __name__ == "__main__":
    dataset = 'qm9'
    atom_list = MOLECULAR_DATASETS[dataset]['atom_list']
    max_atoms = MOLECULAR_DATASETS[dataset]['max_atoms']
    max_types = MOLECULAR_DATASETS[dataset]['max_types']

    # trained model path
    model_path = "results/linesearch/model_checkpoint/qm9/graphspn_zero_sort/dataset=qm9_model=graphspn_zero_sort_nd_n=9_nk_n=5_nk_e=4_nl=3_nr=80_ns=80_ni=40_device=cuda_lr=0.05_betas=[0.9, 0.82]_num_epochs=40_batch_size=256_seed=1_max_atoms=9.pt"

    model = torch.load(model_path)
    torch.manual_seed(2)

    num_samples = 1000
    num_to_show = 8  # assuming at least num_to_show/num_samples is valid
    # nice utility for molecule drawings https://www.rcsb.org/chemical-sketch
    patt_smls = ['C1OCC=C1', 'N1NO1', 'CCCO', 'C1CNC1', 'C1=CC=CC=C1', 'C1CN1C', 'N1C=CC=C1', 'COC']
    cond_smls = []

    for patt in patt_smls:
        xx, aa, submol_size = create_observed_mol(patt, max_atoms, atom_list)
        mols, smls = conditional_sample(model, xx, aa, submol_size, num_samples, max_types, atom_list)
        valid_smls = [sml for mol, sml in zip(mols, smls) if isvalid(mol)]
        valid_mols = [mol for mol in mols if isvalid(mol)]

        # small molecule filtering
        small_smls = [sml for mol, sml in zip(valid_mols[10:], valid_smls[10:]) if len(mol.GetAtoms())-submol_size<submol_size-1]
        final_smls = valid_smls[:8] if len(small_smls) < 3 else valid_smls[:6] + [small_smls[0], small_smls[2]]
        print(len(final_smls))
        print(final_smls)

        cond_smls.append(final_smls)
    
    grid_conditional(cond_smls, patt_smls, useSVG=False)
    grid_unconditional(model, 8, 8, max_atoms, atom_list, useSVG=False)
