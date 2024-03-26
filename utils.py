import re
import copy

from rdkit import Chem
from rdkit.Chem import Descriptors


VALENCY_LIST = {6:4, 7:3, 8:2, 9:1, 15:3, 16:2, 17:1, 35:1, 53:1}

def atom_decoder(atom_list):
    return {i: atom_list[i] for i in range(len(atom_list))}
bond_encoder = {Chem.BondType.SINGLE: 0, Chem.BondType.DOUBLE: 1, Chem.BondType.TRIPLE: 2}
bond_decoder = {0: Chem.BondType.SINGLE, 1: Chem.BondType.DOUBLE, 2: Chem.BondType.TRIPLE}


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

def valency(mol):
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True, None
    except ValueError as error:
        error = str(error)
        i = error.find('#')
        valence = list(map(int, re.findall(r'\d+', error[i:])))
        return False, valence

def valid_mol(mol):
    smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
    mol = Chem.MolFromSmiles(smiles) if mol is not None else None
    if mol is not None and '.' not in Chem.MolToSmiles(mol, isomericSmiles=True):
        return mol
    return None

def evaluate(mols, smiles_gen, smiles_trn, max_mols_gen, return_unique=True, debug=True):
    num_mols = len(mols)

    valid_mols = [valid_mol(mol) for mol in mols]
    valid_mols = [mol for mol in valid_mols if mol is not None]

    smiles_valid = [Chem.MolToSmiles(mol, isomericSmiles=False) for mol in valid_mols]
    smiles_unique = list(set(smiles_valid))

    num_valid_mols = len(valid_mols)
    num_smiles_unique = len(smiles_unique)

    ratio_valid = num_valid_mols / num_mols * 100
    ratio_unique = num_smiles_unique / num_valid_mols * 100 if num_valid_mols > 0 else 0.
    ratio_unique_abs = num_smiles_unique / num_mols * 100

    if return_unique: smiles_valid = smiles_unique
    mols_valid = [Chem.MolFromSmiles(s) for s in smiles_valid]

    if num_mols == 0:
        ratio_novel = 0.
    else:
        novel = num_mols - sum([1 for mol in smiles_gen if mol in smiles_trn])
        ratio_novel = novel / num_mols * 100
        ratio_novel_abs = novel / max_mols_gen * 100


    if debug:
        print("Valid molecules: {}".format(ratio_valid))
        for i, mol in enumerate(valid_mols):
            print("[{}] {}".format(i, Chem.MolToSmiles(mol, isomericSmiles=False)))
    print("Validity: {:.3f}%, Uniqueness: {:.3f}%, Uniqueness (abs): {:.3f}% Novelty: {:.3f}%, Novelty (abs): {:.3f}%".format(
        ratio_valid, ratio_unique, ratio_unique_abs, ratio_novel, ratio_novel_abs))


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
