import os
import torch
import pandas as pd

from rdkit import Chem
from utils.molecular import mols2gs, gs2mols, correct, getvalid


def get_vmols(x, a, atom_list, correct_mols=False, canonical=True):
    if correct_mols == True:
        valid = [correct(mol) for mol in gs2mols(x, a, atom_list)]
    else:
        valid = [getvalid(mol, canonical) for mol in gs2mols(x, a, atom_list)]
    vmols = [mol for mol in valid if mol is not None]
    vsmls = [Chem.MolToSmiles(mol, kekuleSmiles=True, canonical=canonical) for mol in vmols]

    return vmols, vsmls


def metric_v(vmols, num_mols):
    return len(vmols) / num_mols

def metric_u(vsmls, num_mols):
    num_umols = len(list(set(vsmls)))
    if num_umols == 0:
        return 0., 0.
    else:
        return num_umols / len(vsmls), num_umols / num_mols

def metric_n(vsmls, tsmls, num_mols):
    usmls = list(set(vsmls))
    num_umols = len(usmls)
    if num_umols == 0:
        return 0., 0.
    else:
        num_nmols = num_umols - sum([1 for mol in usmls if mol in tsmls])
        return num_nmols / num_umols, num_nmols / num_mols

def metric_s(ratio_v, ratio_u, ratio_n):
    return ratio_v*ratio_u*ratio_n


def evaluate_molecules(x, a, tsmls, atom_list, correct_mols=False, metrics_only=False, canonical=True, affix=''):
    num_mols = len(x)

    vmols, vsmls = get_vmols(x, a, atom_list, correct_mols, canonical)

    ratio_v = metric_v(vmols, num_mols)
    ratio_u, ratio_u_abs = metric_u(vsmls, num_mols)
    ratio_n, ratio_n_abs = metric_n(vsmls, tsmls, num_mols)
    ratio_s = metric_s(ratio_v, ratio_u, ratio_n)

    metrics = {
        f'{affix}valid': ratio_v,
        f'{affix}unique': ratio_u,
        f'{affix}unique_abs': ratio_u_abs,
        f'{affix}novel': ratio_n,
        f'{affix}novel_abs': ratio_n_abs,
        f'{affix}score': ratio_s
    }

    if metrics_only == True:
        return metrics
    else:
        return vmols, vsmls, metrics

def print_metrics(valid, novel, unique, score, novel_abs=[], unique_abs=[], abs=False):
    if abs == True:
        print("V:{:>6.2f}%, U:{:>6.2f}%, U (abs):{:>6.2f}% N:{:>6.2f}%, N (abs):{:>6.2f}%, S:{:>6.2f}%".format(
            100*valid, 100*unique, 100*unique_abs, 100*novel, 100*novel_abs, 100*score))
    else:
        print("V:{:>6.2f}%, U:{:>6.2f}%, N:{:>6.2f}%, S:{:>6.2f}%".format(
            100*valid, 100*unique, 100*novel, 100*score))


def best_model(path):
    files = os.listdir(path)
    dfs = []
    for f in files:
        data = pd.read_csv(path + f)
        data['file_path'] = f.replace('.csv', '.pt')
        dfs.append(data)
    df = pd.concat(dfs, ignore_index=True)
    idx = df['nll_tst_approx'].idxmin()
    return df.loc[idx]['file_path'], df.loc[idx]['nll_val_approx']


def resample_invalid_mols(model, num_samples, atom_list, max_atoms, canonical=True, max_attempts=10):
    n = num_samples
    mols = []

    for _ in range(max_attempts):
        x, a = model.sample(n)
        valid = [getvalid(mol, canonical) for mol in gs2mols(x, a, atom_list)]
        mols.extend([mol for mol in valid if mol is not None])
        n = num_samples - len(mols)
        if len(mols) == num_samples:
            break

    x_valid, a_valid = mols2gs(mols, max_atoms, atom_list)
    if n > 0:
        x_maybe, a_maybe = model.sample(n)
        return torch.cat((x_valid, x_maybe)), torch.cat((a_valid, a_maybe))
    else:
        return x_valid, a_valid

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)




def test_metrics(gsmls, tsmls, max_atom, atom_list, disrupt_mol=None):
    gmols = [Chem.MolFromSmiles(sml) for sml in gsmls]
    [Chem.Kekulize(mol) for mol in gmols]
    x, a = mols2gs(gmols, max_atom, atom_list)
    if disrupt_mol is not None:
        n, c, i, j = disrupt_mol
        a[n, :, i, j] = 0
        a[n, c, i, j] = 1

    metrics = evaluate_molecules(x, a, tsmls, atom_list, correct_mols=False, metrics_only=True)

    print_metrics(metrics['valid'],
                  metrics['novel'],
                  metrics['unique'],
                  metrics['score'],
                  metrics['novel_abs'],
                  metrics['unique_abs'], abs=True)


if __name__ == '__main__':
    # 10 samples from the QM9 dataset
    max_atom = 9
    atom_list = [6, 7, 8, 9, 0]
    tsmls = [
            'CCC1(C)CN1C(C)=O',
            'O=CC1=COC(=O)N=C1',
            'O=CC1(C=O)CN=CO1',
            'CCC1CC2C(O)C2O1',
            'CC1(C#N)C2CN=CN21',
            'CC1(C)OCC1CO',
            'O=C1C=CC2NC2CO1',
            'CC1C=CC(=O)C(C)N1',
            'COCCC1=CC=NN1',
            'CN1C(=O)C2C(O)C21C'
        ]


    gsmls = [
            'CC1(C)CN1C(C)=O',
            'O=CC1=COC(=O)N=C1',
            'O=CC1(C=O)CN=CO1',
            'CCC1CC2C(O)C2O1',
            'CC1(C#N)C2CN=CN21',
            'CC1(C)OCC1CO',
            'O=C1C=CC2NC2CO1',
            'CC1C=CC(=O)C(C)N1',
            'COCCC1=CC=NN1',
            'CN1C(=O)C2C(O)C21C'
        ]
    test_metrics(gsmls, tsmls, max_atom, atom_list, [0, 2, 1, 0])
    test_metrics(gsmls, tsmls, max_atom, atom_list, [4, 2, 1, 0])


    gsmls = [
            'CCC1(C)CN1C(C)=O',
            'CCC1(C)CN1C(C)=O',
            'O=CC1(C=O)CN=CO1',
            'CCC1CC2C(O)C2O1',
            'CC1(C#N)C2CN=CN21',
            'CC1(C)OCC1CO',
            'O=C1C=CC2NC2CO1',
            'CC1C=CC(=O)C(C)N1',
            'COCCC1=CC=NN1',
            'CN1C(=O)C2C(O)C21C'
        ]
    test_metrics(gsmls, tsmls, max_atom, atom_list)


    gsmls = [
            'CC1(C)CN1C(C)=O',
            'O=CC1=COC(=O)N=C1',
            'O=CC1(C=O)CN=CO1',
            'CCC1CC2C(O)C2O1',
            'CC1(C#N)C2CN=CN21',
            'CC1(C)OCC1CO',
            'O=C1C=CC2NC2CO1',
            'CC1C=CC(=O)C(C)N1',
            'COCCC1=CC=NN1',
            'CN1C(=O)C2C(O)C21C'
        ]
    test_metrics(gsmls, tsmls, max_atom, atom_list)
