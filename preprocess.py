import os
import torch
import urllib
import pandas
from rdkit import Chem
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


bond_type_to_int = {Chem.BondType.SINGLE: 0, Chem.BondType.DOUBLE: 1, Chem.BondType.TRIPLE: 2}


class Molecule(object):
    def __init__(self, x, a, n, y):
        self.x = x
        self.a = a
        self.n = n
        self.y = y


class MolecularDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        x = self.data[index].x
        a = self.data[index].a
        n = self.data[index].n
        y = self.data[index].y

        return {'x': x, 'a': a, 'n': n, 'y': y}

    def __len__(self):
        return len(self.data)


def download_qm9():
    file = "qm9_property.csv"
    url = 'https://raw.githubusercontent.com/divelab/DIG_storage/main/ggraph/qm9_property.csv'

    print("Downloading dataset.")

    urllib.request.urlretrieve(url, file)
    data_list = preprocess(file, 'smile', 'penalized_logp', True, 9, [6, 7, 8, 9], fixed_size=True)
    torch.save(MolecularDataset(data_list), "qm9_property.pt")

    os.remove(file)

    print("Done.")


def preprocess(path, smile_col, prop_name, available_prop, num_max_atom, atom_list, fixed_size=True, ohe=False):
    input_df = pandas.read_csv(path, sep=',', dtype='str')
    smile_list = list(input_df[smile_col])
    if available_prop:
        prop_list = list(input_df[prop_name])
    data_list = []

    for i in tqdm(range(len(smile_list))):
        mol = Chem.MolFromSmiles(smile_list[i])
        Chem.Kekulize(mol)
        num_atom = mol.GetNumAtoms()
        if num_atom > num_max_atom:
            continue
        else:
            tensor_size = num_max_atom if fixed_size else num_atom

            if ohe == True:
                atom_tensor = torch.zeros(tensor_size, len(atom_list), dtype=torch.float32)
                for atom_idx, atom in enumerate(mol.GetAtoms()):
                    atom_type = atom.GetAtomicNum()
                    atom_tensor[atom_idx, atom_list.index(atom_type)] = 1
                atom_tensor[~torch.sum(atom_tensor, 1, dtype=torch.bool), 3] = 1

                bond_tensor = torch.zeros(4, tensor_size, tensor_size, dtype=torch.float32)
                for bond in mol.GetBonds():
                    bond_type = bond.GetBondType()
                    c = bond_type_to_int[bond_type]
                    i = bond.GetBeginAtomIdx()
                    j = bond.GetEndAtomIdx()
                    bond_tensor[c, i, j] = 1.0
                    bond_tensor[c, j, i] = 1.0
                bond_tensor[3, ~torch.sum(bond_tensor, 0, dtype=torch.bool)] = 1
            else:
                atom_tensor = torch.zeros(tensor_size, dtype=torch.float32)
                for atom_idx, atom in enumerate(mol.GetAtoms()):
                    atom_type = atom.GetAtomicNum()
                    atom_tensor[atom_idx] = atom_list.index(atom_type) + 1.
                atom_tensor[atom_tensor==0.] = len(atom_list) + 1.

                bond_tensor = torch.zeros(tensor_size, tensor_size, dtype=torch.float32)
                for bond in mol.GetBonds():
                    bond_type = bond.GetBondType()
                    c = bond_type_to_int[bond_type] + 1.
                    i = bond.GetBeginAtomIdx()
                    j = bond.GetEndAtomIdx()
                    bond_tensor[i, j] = c
                    bond_tensor[j, i] = c
                bond_tensor[bond_tensor==0.] = len(bond_type_to_int) + 1.

            if available_prop:
                y = torch.tensor([float(prop_list[i])])
            data_list.append(Molecule(atom_tensor, bond_tensor, num_atom, y))

    return data_list


if __name__ == '__main__':
    # download_qm9()
    dataset = torch.load("qm9_property.pt")
    dataloader = DataLoader(dataset, batch_size=100, shuffle=True)

    for i, x in enumerate(dataloader):
        print(i)
        print(x['x'][0])
        print(x['a'][0])
