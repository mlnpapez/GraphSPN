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

        return x, a, n, y

    def __len__(self):
        return len(self.data)


def download(url, file):
    print("Downloading dataset.")
    urllib.request.urlretrieve(url, file)
    print("Done.")


def preprocess(path, smile_col, prop_name, available_prop, num_max_atom, atom_list, fixed_size=True):
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

            atom_tensor = torch.zeros((tensor_size, len(atom_list)), dtype=torch.float32)
            for atom_idx, atom in enumerate(mol.GetAtoms()):
                atom_type = atom.GetAtomicNum()
                atom_tensor[atom_idx, atom_list.index(atom_type)] = 1
            atom_tensor[~torch.sum(atom_tensor, 1, dtype=torch.bool), 3] = 1

            bond_tensor = torch.zeros([4, tensor_size, tensor_size], dtype=torch.float32)
            for bond in mol.GetBonds():
                bond_type = bond.GetBondType()
                c = bond_type_to_int[bond_type]
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                bond_tensor[c, i, j] = 1.0
                bond_tensor[c, j, i] = 1.0
            bond_tensor[3, ~torch.sum(bond_tensor, 0, dtype=torch.bool)] = 1

            if available_prop:
                y = torch.tensor([float(prop_list[i])])
            data_list.append(Molecule(atom_tensor, bond_tensor, num_atom, y))

    return data_list


def collate_fn(batch):
    return batch


if __name__ == '__main__':
    # url = 'https://raw.githubusercontent.com/divelab/DIG_storage/main/ggraph/qm9_property.csv'
    # download(url, "qm9_property.csv")
    # data_list = preprocess('qm9_property.csv', 'smile', 'penalized_logp', True, 20, [6, 7, 8, 9], fixed_size=True)
    # dataset = MolecularDataset(data_list)
    # torch.save(dataset, "qm9_property.pt")
    dataset = torch.load("qm9_property.pt")
    # dataloader = DataLoader(dataset, batch_size=10, collate_fn=collate_fn, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # x = next(iter(dataloader))
    for i, x in enumerate(dataloader):
        print(i)
        print(x[1][0].permute(1, 2, 0)[5])
