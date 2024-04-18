import os
import re
import copy
import torch
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors
from itertools import compress


VALENCY_LIST = {6:4, 7:3, 8:2, 9:1, 15:3, 16:2, 17:1, 35:1, 53:1}

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

def evaluate(mols, smiles_gen, smiles_trn, max_mols_gen, return_unique=True, debug=False, correct_mols=False):
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

    results = {
        'mols_valid': mols_valid,
        'smiles_valid': smiles_valid,
        'ratio_valid': ratio_valid,
        'ratio_unique': ratio_unique,
        'ratio_unique_abs': ratio_unique_abs,
        'ratio_novel': ratio_novel,
        'ratio_novel_abs': ratio_novel_abs
    }

    return results

def print_results(results, abs=False):
    ratio_valid  = results['ratio_valid']
    ratio_novel  = results['ratio_novel']
    ratio_unique = results['ratio_unique']

    score = ratio_valid*ratio_unique*ratio_novel

    if abs == True:
        ratio_novel_abs  = results['ratio_novel_abs']
        ratio_unique_abs = results['ratio_unique_abs']

        print("Validity: {:.3f}%, Uniqueness: {:.3f}%, Uniqueness (abs): {:.3f}% Novelty: {:.3f}%, Novelty (abs): {:.3f}%, Score: {:.3f}%".format(
            100*ratio_valid, 100*ratio_unique, 100*ratio_unique_abs, 100*ratio_novel, 100*ratio_novel_abs, 100*score))
    else:
        print("Validity: {:.3f}%, Uniqueness: {:.3f}%, Novelty: {:.3f}%, Score: {:.3f}%".format(
            100*ratio_valid, 100*ratio_unique, 100*ratio_novel, 100*score))


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
