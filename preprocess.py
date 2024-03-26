import os
import torch
import urllib
import pandas

from rdkit import Chem
from tqdm import tqdm
from utils import bond_encoder


class MolecularDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def split(self, val_size, tst_size, seed=0):
        generator = torch.Generator()
        generator.manual_seed(seed)
        all_size = len(self.data)

        i_trn, i_val, i_tst = torch.split(torch.randperm(all_size, generator=generator), (all_size-val_size-tst_size, val_size, tst_size))

        data_trn = MolecularDataset([self.data[i] for i in i_trn])
        data_val = MolecularDataset([self.data[i] for i in i_val])
        data_tst = MolecularDataset([self.data[i] for i in i_tst])

        return data_trn, data_val, data_tst


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
                atom_tensor = torch.zeros(tensor_size, len(atom_list), dtype=torch.int8)
                for atom_idx, atom in enumerate(mol.GetAtoms()):
                    atom_type = atom.GetAtomicNum()
                    atom_tensor[atom_idx, atom_list.index(atom_type)] = 1
                atom_tensor[~torch.sum(atom_tensor, 1, dtype=torch.bool), 3] = 1

                bond_tensor = torch.zeros(4, tensor_size, tensor_size, dtype=torch.int8)
                for bond in mol.GetBonds():
                    bond_type = bond.GetBondType()
                    c = bond_encoder[bond_type]
                    i = bond.GetBeginAtomIdx()
                    j = bond.GetEndAtomIdx()
                    bond_tensor[c, i, j] = 1.0
                    bond_tensor[c, j, i] = 1.0
                bond_tensor[3, ~torch.sum(bond_tensor, 0, dtype=torch.bool)] = 1
            else:
                atom_tensor = torch.zeros(tensor_size, dtype=torch.int8)
                for atom_idx, atom in enumerate(mol.GetAtoms()):
                    atom_type = atom.GetAtomicNum()
                    atom_tensor[atom_idx] = atom_list.index(atom_type) + 1
                atom_tensor[atom_tensor==0] = len(atom_list) + 1

                bond_tensor = torch.zeros(tensor_size, tensor_size, dtype=torch.int8)
                for bond in mol.GetBonds():
                    bond_type = bond.GetBondType()
                    c = bond_encoder[bond_type] + 1
                    i = bond.GetBeginAtomIdx()
                    j = bond.GetEndAtomIdx()
                    bond_tensor[i, j] = c
                    bond_tensor[j, i] = c
                bond_tensor[bond_tensor==0] = len(bond_encoder) + 1

            if available_prop:
                y = torch.tensor([float(prop_list[i])])
            data_list.append({'x': atom_tensor-1, 'a': bond_tensor-1, 'n': num_atom, 'y': y, 's': Chem.MolToSmiles(mol)})

    return data_list


def loader_wrapper(x, batch_size, shuffle):
    return torch.utils.data.DataLoader(x, batch_size=batch_size, num_workers=2, shuffle=shuffle, pin_memory=True)

def download_qm9():
    file = "qm9_property.csv"
    url = 'https://raw.githubusercontent.com/divelab/DIG_storage/main/ggraph/qm9_property.csv'

    print("Downloading dataset.")

    urllib.request.urlretrieve(url, file)
    data_list = preprocess(file, 'smile', 'penalized_logp', True, 9, [6, 7, 8, 9], fixed_size=True)
    torch.save(MolecularDataset(data_list), "qm9_property.pt")

    os.remove(file)

    print("Done.")


def load_qm9(batch_size, raw=False, seed=0, val_size=10000, tst_size=10000):
    x = torch.load("qm9_property.pt")
    x_trn, x_val, x_tst = x.split(val_size, tst_size, seed)

    if raw == True:
        return x_trn, x_val, x_tst
    else:
        loader_trn = loader_wrapper(x_trn, batch_size, True)
        loader_val = loader_wrapper(x_val, batch_size, False)
        loader_tst = loader_wrapper(x_tst, batch_size, False)

        return loader_trn, loader_val, loader_tst


if __name__ == '__main__':
    # download_qm9()
    loader_trn, loader_val, loader_tst = load_qm9(100)

    for i, x in enumerate(loader_trn):
        print(i)
        print(x['x'][0])
        print(x['a'][0])
        print(x['s'][0])

    x_trn, x_val, x_tst = load_qm9(0, raw=True)

    smiles = [x['s'] for x in x_trn]
    print(smiles)
