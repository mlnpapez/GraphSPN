import torch
import graphspn
import graphspn_zero
import datasets
from utils import *
from plot_utils import *

from rdkit import Chem, rdBase
from rdkit.Chem import Draw

rdBase.DisableLog("rdApp.error")

def conditional_sample(model, xx, aa, submol_size, num_samples):
    # NOTE: one obs only
    mol_size = xx.shape[-1]
    marginalize(model.network, model.nd_nodes, mol_size-submol_size, submol_size)
    z = flatten_graph(xx, aa)

    z = z.to(model.device)
    with torch.no_grad():
        z = z.expand(num_samples, -1)
        sample = model.network.sample(x=z.to(torch.float)).cpu()
        x_new, a_new = unflatt_graph(sample, model.nd_nodes, model.nd_nodes)
        mols, smls = create_mols(x_new, a_new, model.atom_list)

    return mols, smls

def create_observed_mol(smile='C1OCC=C1'):
    dataset_info = datasets.MOLECULAR_DATASETS['qm9']
    mol = Chem.MolFromSmiles(smile)
    xx, aa = mol_to_graph(mol, dataset_info['nd'], dataset_info['atom_list'])
    mol_size = len(mol.GetAtoms())
    return xx.unsqueeze(0), aa.unsqueeze(0), mol_size
    
def sample_and_filter(model):
    pass

if __name__ == "__main__":
    model_path = "results/training/model_checkpoint/qm9/graphspn_zero_sort/dataset=qm9_model=graphspn_zero_sort_nd_n=9_nk_n=5_nk_e=4_nl=2_nr=40_ns=40_ni=40_device=cuda_optimizer=adam_lr=0.05_betas=[0.9, 0.82]_num_epochs=20_batch_size=1000_seed=0.pt"

    model = torch.load(model_path)
    torch.manual_seed(0)

    num_samples = 100
    num_to_show = 8  # assuming at least num_to_show/num_samples is valid
    # nice utility for molecule drawings https://www.rcsb.org/chemical-sketch
    patt_smls = ['C1OCC=C1', 'N1NO1', 'CCCO', 'C1CNC1']
    cond_smls = []

    for patt in patt_smls:
        xx, aa, submol_size = create_observed_mol(patt)
        mols, smls = conditional_sample(model, xx, aa, submol_size, num_samples)
        valid_smls = [sml for mol, sml in zip(mols, smls) if isvalid(mol)][:num_to_show]
        cond_smls.append(valid_smls)
    
    highlight_grid(cond_smls, patt_smls)

