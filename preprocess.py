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
    input_df = pandas.read_csv(f'{path}.csv', sep=',', dtype='str')
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

                bond_tensor = torch.zeros(tensor_size, tensor_size, 4, dtype=torch.int8)
                for bond in mol.GetBonds():
                    bond_type = bond.GetBondType()
                    c = bond_encoder[bond_type]
                    i = bond.GetBeginAtomIdx()
                    j = bond.GetEndAtomIdx()
                    bond_tensor[i, j, c] = 1.0
                    bond_tensor[j, i, c] = 1.0
                bond_tensor[~torch.sum(bond_tensor, 0, dtype=torch.bool), 3] = 1

                atom_tensor_deq = atom_tensor + torch.rand(tensor_size, len(atom_list))
                bond_tensor_deq = bond_tensor + torch.rand(tensor_size, tensor_size, 4)

                if available_prop:
                    y = torch.tensor([float(prop_list[i])])
                    data_list.append({'x': atom_tensor,
                                      'a': bond_tensor,
                                      'n': num_atom,
                                      'y': y,
                                      's': Chem.MolToSmiles(mol),
                                      'x_deq': atom_tensor_deq,
                                      'a_deq': bond_tensor_deq})
                else:
                    data_list.append({'x': atom_tensor,
                                      'a': bond_tensor,
                                      'n': num_atom,
                                      's': Chem.MolToSmiles(mol),
                                      'x_deq': atom_tensor_deq,
                                      'a_deq': bond_tensor_deq})

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

                atom_tensor -= 1
                bond_tensor -= 1

                if available_prop:
                    y = torch.tensor([float(prop_list[i])])
                    data_list.append({'x': atom_tensor,
                                      'a': bond_tensor,
                                      'n': num_atom,
                                      'y': y,
                                      's': Chem.MolToSmiles(mol)})
                else:
                    data_list.append({'x': atom_tensor,
                                      'a': bond_tensor,
                                      'n': num_atom,
                                      's': Chem.MolToSmiles(mol)})

    if ohe == True:
        torch.save(MolecularDataset(data_list), f'{path}_ohe.pt')
    else:
        torch.save(MolecularDataset(data_list), f'{path}_int.pt')


def loader_wrapper(x, batch_size, shuffle):
    return torch.utils.data.DataLoader(x, batch_size=batch_size, num_workers=2, shuffle=shuffle, pin_memory=True)

def download_qm9(ohe=False):
    file = 'qm9_property'
    url = 'https://raw.githubusercontent.com/divelab/DIG_storage/main/ggraph/qm9_property.csv'

    print('Downloading and preprocessing dataset.')

    urllib.request.urlretrieve(url, f'{file}.csv')
    preprocess(file, 'smile', 'penalized_logp', True, 9, [6, 7, 8, 9], fixed_size=True, ohe=ohe)
    os.remove(f'{file}.csv')

    print('Done.')


def load_qm9(batch_size, raw=False, seed=0, val_size=10000, tst_size=10000, ohe=False):
    if ohe == True:
        x = torch.load('qm9_property_ohe.pt')
    else:
        x = torch.load('qm9_property_int.pt')
    x_trn, x_val, x_tst = x.split(val_size, tst_size, seed)

    if raw == True:
        return x_trn, x_val, x_tst
    else:
        loader_trn = loader_wrapper(x_trn, batch_size, True)
        loader_val = loader_wrapper(x_val, batch_size, False)
        loader_tst = loader_wrapper(x_tst, batch_size, False)

        return loader_trn, loader_val, loader_tst


if __name__ == '__main__':
    ohe = True
    download_qm9(ohe)
    loader_trn, loader_val, loader_tst = load_qm9(100, ohe=ohe)

    for i, x in enumerate(loader_trn):
        print(i)
        print(x['x'][0])
        print(x['a'][0])
        print(x['s'][0])
        if ohe == True:
            print(x['x_deq'][0])
            print(x['a_deq'][0])

    x_trn, x_val, x_tst = load_qm9(0, raw=True, ohe=ohe)

    smiles = [x['s'] for x in x_trn]
    print(smiles)
