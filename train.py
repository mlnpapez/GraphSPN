import json
import torch

from rdkit import RDLogger
from utils.datasets import MOLECULAR_DATASETS, load_dataset
from utils.train import train, evaluate

from models import graphspn_prel
from models import graphspn_zero
from models import graphspn_marg
from models import graphspn_back
from models import moflow

MODELS = {**graphspn_prel.MODELS,
          **graphspn_zero.MODELS,
          **graphspn_marg.MODELS,
          **graphspn_back.MODELS,
          'moflow': moflow.MoFlow,}


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    RDLogger.DisableLog('rdApp.*')

    dataset = 'qm9'
    names = ['graphspn_zero_none'] # MODELS.keys()

    checkpoint_dir = 'results/training/model_checkpoint/'
    trainepoch_dir = 'results/training/model_trainepoch/'
    evaluation_dir = 'results/training/model_evaluation/'
    hyperparam_dir = 'configs/training/model_hyperparam/'

    for name in names:
        with open(f'config/{dataset}/{name}.json', 'r') as f:
            hyperpars = json.load(f)

        model = MODELS[name](**hyperpars['model_hyperpars'])
        print(model)

        hyperpars['atom_list'] = MOLECULAR_DATASETS[dataset]['atom_list']
        hyperpars['max_atoms'] = MOLECULAR_DATASETS[dataset]['max_atoms']

        loader_trn, loader_val, loader_tst = load_dataset(hyperpars['dataset'], hyperpars['batch_size'], split=[0.8, 0.1, 0.1])
        # loader_trn, smiles_trn = permute_dataset(loader_trn, dataset, permutation='canonical')
        smiles_trn = [x['s'] for x in loader_trn.dataset]

        path = train(model, loader_trn, loader_val, smiles_trn, hyperpars, checkpoint_dir)
        model = torch.load(path)
        metrics = evaluate(model, loader_trn, loader_val, loader_tst, smiles_trn, hyperpars, evaluation_dir, compute_nll=False)

        print("\n".join(f'{key:<16}{value:>10.4f}' for key, value in metrics.items()))
