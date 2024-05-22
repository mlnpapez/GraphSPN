import os
import re
import copy
import torch
import pandas as pd
import torch.optim as optim

from rdkit import Chem
from rdkit.Chem import Descriptors
from itertools import compress
from rdkit.Chem.Draw import MolsToGridImage
from tqdm import tqdm


IGNORED_HYPERPARS = [
    'atom_list'
]

VALENCY_LIST = {6:4, 7:3, 8:2, 9:1, 15:3, 16:2, 17:1, 35:1, 53:1}

def flatten_dict(d, input_key=''):
    if isinstance(d, dict):
        return {k if input_key else k: v for key, value in d.items() for k, v in flatten_dict(value, key).items()}
    else:
        return {input_key: d}

def dict2str(d):
    return '_'.join([f'{key}={value}' for key, value in d.items() if key not in IGNORED_HYPERPARS])


def atom_decoder(atom_list):
    return {i: atom_list[i] for i in range(len(atom_list))}
bond_encoder = {Chem.BondType.SINGLE: 0, Chem.BondType.DOUBLE: 1, Chem.BondType.TRIPLE: 2}
bond_decoder = {0: Chem.BondType.SINGLE, 1: Chem.BondType.DOUBLE, 2: Chem.BondType.TRIPLE}

def valency(mol):
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True, None
    except ValueError as error:
        error = str(error)
        i = error.find('#')
        valence = list(map(int, re.findall(r'\d+', error[i:])))
        return False, valence

def correct_mol(mol):
    while True:
        flag, atomid_valence = valency(mol)
        if flag:
            break
        else:
            assert len (atomid_valence) == 2
            queue = []
            for b in mol.GetAtomWithIdx(atomid_valence[0]).GetBonds():
                queue.append((b.GetIdx(), int(b.GetBondType()), b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
            queue.sort(key=lambda tup: tup[1], reverse=True)
            if len(queue) > 0:
                start = queue[0][2]
                end = queue[0][3]
                bond_index = queue[0][1]
                mol.RemoveBond(start, end)
                if bond_index < 3:
                    mol.AddBond(start, end, bond_decoder[bond_index])

    return mol

def radical_electrons_to_hydrogens(mol):
    mol = copy.deepcopy(mol)
    if Descriptors.NumRadicalElectrons(mol) == 0:
        return mol
    else:
        print('Converting radical electrons to hydrogens.')
        for a in mol.GetAtoms():
            num_radicals = a.GetNumRadicalElectrons()
            if num_radicals > 0:
                a.SetNumRadicalElectrons(0)
                a.SetNumExplicitHs(num_radicals)
    return mol

def valid_mol(mol):
    smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
    mol = Chem.MolFromSmiles(smiles) if mol is not None else None
    if mol is not None and '.' not in Chem.MolToSmiles(mol, isomericSmiles=True):
        return mol
    return None

def isvalid(mol):
    smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
    mol = Chem.MolFromSmiles(smiles) if mol is not None else None
    if mol is not None and '.' not in Chem.MolToSmiles(mol, isomericSmiles=True):
        return True
    else:
        return False

def create_mols(x, a, atom_list):
    nd_nodes = x.size(1)
    mols = []
    smls = []
    for x, a in zip(x, a):
        rw_mol = Chem.RWMol()

        for i in range(nd_nodes):
            if x[i].item() < len(atom_list):
                rw_mol.AddAtom(Chem.Atom(atom_decoder(atom_list)[x[i].item()]))

        num_atoms = rw_mol.GetNumAtoms()

        for i in range(num_atoms):
            for j in range(num_atoms):
                if a[i, j].item() < 3 and i > j:
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

        rw_mol = radical_electrons_to_hydrogens(rw_mol)

        mols.append(rw_mol)
        smls.append(Chem.MolToSmiles(rw_mol))

    return mols, smls


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

def permute_graph(xx, aa, pi):
    px = xx[:, pi]
    pa = aa[:, pi, :]
    pa = pa[:, :, pi]
    return px, pa

def flatten_graph(xx, aa, dim=2):
    n = xx.shape[1]
    z = torch.cat((xx.unsqueeze(dim), aa), dim=dim)
    return z.view(-1, n + n**2)

def unflatt_graph(z, nd_nodes, num_full):
    z = z.view(-1, nd_nodes, nd_nodes+1)
    x = z[:, 0:num_full, 0 ]
    a = z[:, 0:num_full, 1:num_full+1]
    return x, a


def evaluate_molecules(mols, smiles_gen, smiles_trn, max_mols_gen, return_unique=True, debug=False, correct_mols=False, metrics_only=False, affix=''):
    num_mols = len(mols)

    if correct_mols == True:
        valid_mols = [correct_mol(mol) for mol in mols]
    else:
        valid_mols = [valid_mol(mol) for mol in mols]
    valid_mols = [mol for mol in valid_mols if mol is not None]

    smiles_valid = [Chem.MolToSmiles(mol, isomericSmiles=False) for mol in valid_mols]
    smiles_unique = list(set(smiles_valid))

    num_valid_mols = len(valid_mols)
    num_smiles_unique = len(smiles_unique)

    ratio_valid = num_valid_mols / num_mols
    ratio_unique = num_smiles_unique / num_valid_mols if num_valid_mols > 0 else 0.
    ratio_unique_abs = num_smiles_unique / num_mols

    if return_unique == True: smiles_valid = smiles_unique
    mols_valid = [Chem.MolFromSmiles(s) for s in smiles_valid]

    if num_mols == 0:
        ratio_novel = 0.
    else:
        novel = num_mols - sum([1 for mol in smiles_gen if mol in smiles_trn])
        ratio_novel = novel / num_mols
        ratio_novel_abs = novel / max_mols_gen

    if debug == True:
        print("Valid molecules: {}".format(ratio_valid))
        for i, mol in enumerate(valid_mols):
            print("[{}] {}".format(i, Chem.MolToSmiles(mol, isomericSmiles=False)))

    metrics = {
        f'{affix}valid': ratio_valid,
        f'{affix}unique': ratio_unique,
        f'{affix}unique_abs': ratio_unique_abs,
        f'{affix}novel': ratio_novel,
        f'{affix}novel_abs': ratio_novel_abs,
        f'{affix}score': ratio_valid*ratio_unique*ratio_novel
    }

    if metrics_only == True:
        return metrics
    else:
        return mols_valid, smiles_valid, metrics

def print_metrics(valid, novel, unique, score, novel_abs=[], unique_abs=[], abs=False):
    if abs == True:
        print("Validity: {:.3f}%, Uniqueness: {:.3f}%, Uniqueness (abs): {:.3f}% Novelty: {:.3f}%, Novelty (abs): {:.3f}%, Score: {:.3f}%".format(
            100*valid, 100*unique, 100*unique_abs, 100*novel, 100*novel_abs, 100*score))
    else:
        print("Validity: {:.3f}%, Uniqueness: {:.3f}%, Novelty: {:.3f}%, Score: {:.3f}%".format(
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


def resample_invalid_mols(model, num_samples):
    n = num_samples
    mols = []
    smls = []

    while len(mols) != num_samples:
        mols_gen, smls_gen = model.sample(n)
        mols_val = [isvalid(mol) for mol in mols_gen]
        mols.extend(compress(mols_gen, mols_val))
        smls.extend(compress(smls_gen, mols_val))
        n = num_samples - len(mols)

    return mols, smls

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_epoch(model, loader, optimizer=[], verbose=False):
    nll_sum = 0.
    for x in tqdm(loader, leave=False, disable=verbose):
        nll = -model.logpdf(x)
        nll_sum += nll * len(x)
        if optimizer:
            optimizer.zero_grad()
            nll.backward()
            optimizer.step()

    return (nll_sum / len(loader.dataset)).item()

METRIC_TYPES = ['valid', 'unique', 'novel', 'score']

def train(model,
          loader_trn,
          loader_val,
          smiles_trn,
          hyperpars,
          checkpoint_dir,
          trainepoch_dir,
          num_nonimproving_epochs=30,
          verbose=False,
          metric_type='score'
    ):
    optimizer = optim.Adam(model.parameters(), **hyperpars['optimizer_hyperpars'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
 
    lookahead_counter = num_nonimproving_epochs
    if metric_type in METRIC_TYPES:
        best_metric = 0e0
    else:
        best_metric = 1e6
    best_model_path = None
    save_model = False

    for epoch in range(hyperpars['num_epochs']):
        model.train()
        nll_trn = run_epoch(model, loader_trn, verbose=verbose, optimizer=optimizer)
        scheduler.step()
        model.eval()

        molecules_sam, smiles_sam = model.sample(200)
        metrics = evaluate_molecules(molecules_sam, smiles_sam, smiles_trn, 200, metrics_only=True)
        metrics_str = f'valid={metrics["valid"]:.2f}, unique={metrics["unique"]:.2f}, novel={metrics["novel"]:.2f}, score={metrics["score"]:.2f}'

        if metric_type in METRIC_TYPES:
            metric = metrics[metric_type]
            print(f'epoch {epoch:3d}: ll_trn={-nll_trn:.4f}, ' + metrics_str)

            if metric > best_metric:
                best_metric = metric
                lookahead_counter = num_nonimproving_epochs
                save_model = True
            else:
                lookahead_counter -= 1
        else:
            metric = run_epoch(model, loader_val, verbose=verbose)
            print(f'epoch {epoch:3d}: ll_trn={-nll_trn:.4f}, ll_val={-metric:.4f}, ' + metrics_str)

            if metric < best_metric:
                best_metric = metric
                lookahead_counter = num_nonimproving_epochs
                save_model = True
            else:
                lookahead_counter -= 1

        if lookahead_counter == 0:
            break

        if save_model == True:
            dir = checkpoint_dir + f'{hyperpars["dataset"]}/{hyperpars["model"]}/'

            if os.path.isdir(dir) != True:
                os.makedirs(dir)
            if best_model_path != None:
                os.remove(best_model_path)
            path = dir + dict2str(flatten_dict(hyperpars)) + '.pt'
            torch.save(model, path)
            best_model_path = path
            save_model == False

    return best_model_path

def evaluate(model, loader_trn, loader_val, loader_tst, smiles_trn, hyperpars, evaluation_dir, num_samples=1000, compute_nll=True):
    model.eval()

    molecules_sam, smiles_sam = model.sample(num_samples)
    molecules_res, smiles_res = resample_invalid_mols(model, num_samples)

    molecules_res_f, _, metrics_resample_f = evaluate_molecules(molecules_sam, smiles_sam, smiles_trn, num_samples, correct_mols=False, affix='res_f_')
    molecules_res_t, _, metrics_resample_t = evaluate_molecules(molecules_res, smiles_res, smiles_trn, num_samples, correct_mols=False, affix='res_t_')
    molecules_cor_t, _, metrics_correction = evaluate_molecules(molecules_sam, smiles_res, smiles_trn, num_samples, correct_mols=True,  affix='cor_t_')

    if compute_nll == True:
        nll_trn_approx = run_epoch(model, loader_trn)
        nll_val_approx = run_epoch(model, loader_val)
        nll_tst_approx = run_epoch(model, loader_tst)
        metrics_neglogliks = {
            'nll_trn_approx': nll_trn_approx,
            'nll_val_approx': nll_val_approx,
            'nll_tst_approx': nll_tst_approx
        }
    else:
        metrics_neglogliks = {}

    metrics = {**metrics_resample_f, **metrics_resample_t, **metrics_correction, **metrics_neglogliks, "num_params": count_parameters(model)}

    dir = evaluation_dir + f'metrics/{hyperpars["dataset"]}/{hyperpars["model"]}/'
    if os.path.isdir(dir) != True:
        os.makedirs(dir)
    path = dir + dict2str(flatten_dict(hyperpars))
    df = pd.DataFrame.from_dict({**flatten_dict(hyperpars), **metrics}, 'index').transpose()
    df.to_csv(path + '.csv', index=False)

    dir = evaluation_dir + f'images/{hyperpars["dataset"]}/{hyperpars["model"]}/'
    if os.path.isdir(dir) != True:
        os.makedirs(dir)
    path = dir + dict2str(flatten_dict(hyperpars))

    img_res_f = MolsToGridImage(mols=molecules_res_f[0:64], molsPerRow=8, subImgSize=(200, 200), useSVG=False)
    img_res_t = MolsToGridImage(mols=molecules_res_t[0:64], molsPerRow=8, subImgSize=(200, 200), useSVG=False)
    img_cor_t = MolsToGridImage(mols=molecules_cor_t[0:64], molsPerRow=8, subImgSize=(200, 200), useSVG=False)

    img_res_f.save(path + f'_img_res_f.png')
    img_res_t.save(path + f'_img_res_t.png')
    img_cor_t.save(path + f'_img_cor_t.png')

    return metrics
