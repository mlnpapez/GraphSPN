import torch
import graphspn_naive
import graphspn_zero
import datasets
from utils import *
from plot_utils import *

from rdkit import Chem, rdBase

rdBase.DisableLog("rdApp.error")

def conditional_sample(model, xx, aa, submol_size, num_samples):
    # NOTE: one obs only
    max_size = xx.shape[-1]
    marginalize(model.network, model.nd_nodes, max_size-submol_size, submol_size)
    z = flatten_graph(xx, aa)

    z = z.to(model.device)
    with torch.no_grad():
        z = z.expand(num_samples, -1)
        sample = model.network.sample(x=z.to(torch.float)).cpu()
        xx_sample, aa_sample = unflatt_graph(sample, model.nd_nodes, model.nd_nodes)
        mol_sample, sml_sample = create_mols(xx_sample.int(), aa_sample.int(), model.atom_list)

    return mol_sample, sml_sample

def create_observed_mol(smile='C1OCC=C1', dataset_name='qm9'):
    dataset_info = datasets.MOLECULAR_DATASETS[dataset_name]
    mol = Chem.MolFromSmiles(smile)
    xx, aa = mol_to_graph(mol, dataset_info['nd'], dataset_info['atom_list'])
    mol_size = len(mol.GetAtoms())  # number of atoms in molecule
    return xx.unsqueeze(0), aa.unsqueeze(0), mol_size

if __name__ == "__main__":
    model_path = "results/training/model_checkpoint/qm9/graphspn_zero_sort/dataset=qm9_model=graphspn_zero_sort_nd_n=9_nk_n=5_nk_e=4_nl=2_nr=40_ns=40_ni=40_device=cuda_optimizer=adam_lr=0.05_betas=[0.9, 0.82]_num_epochs=20_batch_size=1000_seed=0.pt"

    model = torch.load(model_path)
    torch.manual_seed(1)

    num_samples = 2000
    num_to_show = 7  # assuming at least num_to_show/num_samples is valid
    # nice utility for molecule drawings https://www.rcsb.org/chemical-sketch
    patt_smls = ['C1OCC=C1', 'N1NO1', 'CCCO', 'C1CNC1']
    cond_smls = []

    for patt in patt_smls:
        xx, aa, submol_size = create_observed_mol(patt)
        mols, smls = conditional_sample(model, xx, aa, submol_size, num_samples)
        valid_smls = [sml for mol, sml in zip(mols, smls) if isvalid(mol)]
        valid_mols = [mol for mol in mols if isvalid(mol)]

        # small molecule filtering
        small_smls = [sml for mol, sml in zip(valid_mols[5:], valid_smls[5:]) if len(mol.GetAtoms())-submol_size<submol_size-1]
        final_smls = valid_smls[:7] if len(small_smls) < 3 else valid_smls[:5] + [small_smls[0], small_smls[2]]
        print(final_smls)

        cond_smls.append(final_smls)
    
    highlight_grid(cond_smls, patt_smls, useSVG=True)
    joint_grid(model, 4, 7, useSVG=True)

