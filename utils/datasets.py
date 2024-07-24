import os
import torch
import urllib
import pandas

from rdkit import Chem
from tqdm import tqdm

from utils.molecular import mol2g


MOLECULAR_DATASETS = {
    'qm9': {
        'dataset': 'qm9',
        'max_types': 5,
        'max_atoms': 9,
        'atom_list': [6, 7, 8, 9, 0]
    },
    'zinc250k': {
        'dataset': 'zinc250k',
        'max_types': 10,
        'max_atoms': 38,
        'atom_list': [6, 7, 8, 9, 15, 16, 17, 35, 53, 0]
    }
}


def download_qm9(dir='data/'):
    if os.path.isdir(dir) != True:
        os.makedirs(dir)

    file = f'{dir}qm9'
    url = 'https://raw.githubusercontent.com/divelab/DIG_storage/main/ggraph/qm9_property.csv'

    print('Downloading and preprocessing the QM9 dataset.')

    urllib.request.urlretrieve(url, f'{file}.csv')
    preprocess(file, 'smile', 'penalized_logp', MOLECULAR_DATASETS['qm9']['nd'], MOLECULAR_DATASETS['qm9']['atom_list'])
    os.remove(f'{file}.csv')

    print('Done.')

def download_zinc250k(dir='data/'):
    if os.path.isdir(dir) != True:
        os.makedirs(dir)

    file = f'{dir}zinc250k'
    url = 'https://raw.githubusercontent.com/divelab/DIG_storage/main/ggraph/zinc250k_property.csv'

    print('Downloading and preprocessing the Zinc250k dataset.')

    urllib.request.urlretrieve(url, f'{file}.csv')
    preprocess(file, 'smile', 'penalized_logp', MOLECULAR_DATASETS['zinc250k']['nd'], MOLECULAR_DATASETS['zinc250k']['atom_list'])
    os.remove(f'{file}.csv')

    print('Done.')


def preprocess(path, smile_col, prop_name, max_atom, atom_list):
    input_df = pandas.read_csv(f'{path}.csv', sep=',', dtype='str')
    smls_list = list(input_df[smile_col])
    prop_list = list(input_df[prop_name])
    data_list = []

    for smls, prop in tqdm(zip(smls_list, prop_list)):
        mol = Chem.MolFromSmiles(smls)
        Chem.Kekulize(mol)

        x, a = mol2g(mol, max_atom, atom_list)
        y = torch.tensor([float(prop)])

        data_list.append({'x': x, 'a': a, 'n': mol.GetNumAtoms(), 'y': y, 's': Chem.MolToSmiles(mol, kekuleSmiles=True)})

    torch.save(data_list, f'{path}.pt')

class DictDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

def load_dataset(name, batch_size, raw=False, seed=0, split=None, dir='data/'):
    x = DictDataset(torch.load(f'{dir}{name}.pt'))

    if split is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
        x_trn, x_val, x_tst = torch.utils.data.random_split(x, split, generator=generator)

        if raw == True:
            return x_trn, x_val, x_tst
        else:
            loader_trn = torch.utils.data.DataLoader(x_trn, batch_size=batch_size, num_workers=2, shuffle=True,  pin_memory=True)
            loader_val = torch.utils.data.DataLoader(x_val, batch_size=batch_size, num_workers=2, shuffle=False, pin_memory=True)
            loader_tst = torch.utils.data.DataLoader(x_tst, batch_size=batch_size, num_workers=2, shuffle=False, pin_memory=True)

            return loader_trn, loader_val, loader_tst
    else:
        if raw == True:
            return x
        else:
            return torch.utils.data.DataLoader(x, batch_size=batch_size, num_workers=2, shuffle=True,  pin_memory=True)


if __name__ == '__main__':
    download = False
    dataset = 'qm9'
    if download:
        if dataset == 'qm9':
            download_qm9()
        elif dataset == 'zinc250k':
            download_zinc250k()
        else:
            os.error('Unsupported dataset.')

    loader_trn, loader_val, loader_tst = load_dataset(dataset, 100, split=[0.8, 0.1, 0.1])

    for i, x in enumerate(loader_trn):
        print(i)
        print(x['x'][0])
        print(x['a'][0])
        print(x['s'][0])

    # x_trn, x_val, x_tst = load_dataset(dataset, 0, raw=True)

    # smiles = [x['s'] for x in x_trn]
    # print(smiles)




# def permute_dataset(loader, dataset, permutation='all'):
#     nd = MOLECULAR_DATASETS[dataset]['nd']
#     al = MOLECULAR_DATASETS[dataset]['atom_list']

#     single = torch.randperm(nd)

#     smls = []
#     for d in loader.dataset.data:
#         xx, aa = d['x'], d['a']

#         num_full = torch.sum(xx != len(al))
#         if permutation == 'all':
#             pi = torch.cat((torch.randperm(num_full), torch.arange(num_full, nd)))
#         elif permutation == 'single':
#             pi = torch.cat((single[0:num_full], torch.arange(num_full, nd)))
#         elif permutation == 'canonical':
#             pi = torch.arange(nd) # TODO: This is not very efficient.

#         px = xx[pi]
#         pa = aa[pi, :]
#         pa = pa[:, pi]

#         d['x'], d['a'] = px, pa
#         smls.append(Chem.MolToSmiles(graph_to_mol(px, pa, al))) # canonical=True ?

#     return loader, smls
