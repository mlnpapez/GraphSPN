import re
import torch

from rdkit import Chem


VALENCY_LIST = {6:4, 7:3, 8:2, 9:1, 15:3, 16:2, 17:1, 35:1, 53:1}

BOND_ENCODER = {Chem.BondType.SINGLE: 0, Chem.BondType.DOUBLE: 1, Chem.BondType.TRIPLE: 2}
BOND_DECODER = {0: Chem.BondType.SINGLE, 1: Chem.BondType.DOUBLE, 2: Chem.BondType.TRIPLE}


def mol2x(mol, max_atom, atom_list):
    num_atom = len(atom_list)
    atom_tensor = torch.zeros(max_atom, num_atom, dtype=torch.int8)
    for atom_idx, atom in enumerate(mol.GetAtoms()):
        atom_tensor[atom_idx, atom_list.index(atom.GetAtomicNum())] = 1
    atom_tensor[~torch.sum(atom_tensor, 1, dtype=torch.bool), num_atom-1] = 1
    return atom_tensor

def mol2a(mol, max_atom):
    bond_tensor = torch.zeros(4, max_atom, max_atom, dtype=torch.int8)
    for bond in mol.GetBonds():
        c = BOND_ENCODER[bond.GetBondType()]
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_tensor[c, i, j] = 1.0
        bond_tensor[c, j, i] = 1.0
    bond_tensor[3, ~torch.sum(bond_tensor, 0, dtype=torch.bool)] = 1
    return bond_tensor

def mol2g(mol, max_atom, atom_list):
    x = mol2x(mol, max_atom, atom_list)
    a = mol2a(mol, max_atom)
    return x, a

def g2mol(x, a, atom_list):
    mol = Chem.RWMol()

    atoms = torch.argmax(x, axis=1)
    atoms_exist = atoms != len(atom_list) - 1
    atoms = atoms[atoms_exist]

    for atom in atoms:
        mol.AddAtom(Chem.Atom(atom_list[atom]))

    bonds = torch.argmax(a, axis=0)
    bonds = bonds[atoms_exist, :][:, atoms_exist]
    for start, end in zip(*torch.nonzero(bonds < 3, as_tuple=True)):
        if start > end:
            mol.AddBond(int(start), int(end), BOND_DECODER[bonds[start, end].item()])
            flag, valence = valency(mol)
            if flag:
                continue
            else:
                assert len(valence) == 2
                i = valence[0]
                v = valence[1]
                a = mol.GetAtomWithIdx(i).GetAtomicNum()
                if a in (7, 8, 16) and (v - VALENCY_LIST[a]) == 1:
                    mol.GetAtomWithIdx(i).SetFormalCharge(1)
    return mol

def mols2gs(mols, max_atom, atom_list):
    x = torch.stack([mol2x(mol, max_atom, atom_list) for mol in mols])
    a = torch.stack([mol2a(mol, max_atom) for mol in mols])
    return x, a

def gs2mols(x, a, atom_list):
    return [g2mol(x, a, atom_list) for x, a in zip(x, a)]


def valency(mol):
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True, None
    except ValueError as error:
        error = str(error)
        i = error.find('#')
        valence = list(map(int, re.findall(r'\d+', error[i:])))
        return False, valence

def correct(mol):
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
                    mol.AddBond(start, end, BOND_DECODER[bond_index])

    return mol

# def getvalid(mol, canonical=True):
#     _mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol, canonical=canonical)) if mol is not None else None
#     if _mol is not None and '.' not in Chem.MolToSmiles(_mol, canonical=canonical):
#         Chem.Kekulize(_mol)
#         return _mol
#     else:
#         return None

def getvalid(mol, canonical=True):
    sml = Chem.MolToSmiles(mol, canonical=canonical)
    if Chem.MolFromSmiles(sml) is not None and mol is not None and '.' not in sml:
        return mol
    else:
        return None

# def isvalid(mol, canonical=True):
#     _mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol, canonical=canonical)) if mol is not None else None
#     if _mol is not None and '.' not in Chem.MolToSmiles(_mol, canonical=canonical):
#         return True
#     else:
#         return False

def isvalid(mol, canonical=True):
    sml = Chem.MolToSmiles(mol, canonical=canonical)
    if Chem.MolFromSmiles(sml) is not None and mol is not None and '.' not in sml:
        return True
    else:
        return False


if __name__ == '__main__':
    # 10 samples from the QM9 dataset
    max_atom = 9
    atom_list = [6, 7, 8, 9, 0]
    smiles = [
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

    for sa in smiles:
        mol = Chem.MolFromSmiles(sa)
        Chem.Kekulize(mol)
        x, a = mol2g(mol, max_atom, atom_list)
        mol = g2mol(x, a, atom_list)
        sb = Chem.MolToSmiles(mol, kekuleSmiles=True)
        print(sa)
        print(sb)
