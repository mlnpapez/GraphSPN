import json
import torch

from rdkit import RDLogger
from utils.datasets import MOLECULAR_DATASETS, load_dataset
from utils.train import train, evaluate
from utils.evaluate import count_parameters

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


CHECKPOINT_DIR = 'results/training/model_checkpoint/'
EVALUATION_DIR = 'results/training/model_evaluation/'


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    RDLogger.DisableLog('rdApp.*')

    dataset = 'zinc250k'
    names = ['graphspn_back_none'] # MODELS.keys()

    for name in names:
        with open(f'config/{dataset}/{name}.json', 'r') as f:
            hyperpars = json.load(f)
        hyperpars['atom_list'] = MOLECULAR_DATASETS[dataset]['atom_list']
        hyperpars['max_atoms'] = MOLECULAR_DATASETS[dataset]['max_atoms']

        model = MODELS[name](**hyperpars['model_hyperpars'])
        print(dataset)
        print(json.dumps(hyperpars, indent=4))
        print(model)
        print(f'The number of parameters is {count_parameters(model)}.')

        if 'sort' in name:
            canonical = True
        else:
            canonical = True

        loader_trn, loader_val = load_dataset(hyperpars['dataset'], hyperpars['batch_size'], split=None, canonical=canonical)
        smiles_trn = [x['s'] for x in loader_trn.dataset]

        path = train(model, loader_trn, loader_val, smiles_trn, hyperpars, CHECKPOINT_DIR)
        model = torch.load(path)
        metrics = evaluate(model, loader_trn, loader_val, smiles_trn, hyperpars, EVALUATION_DIR, compute_nll=False, canonical=canonical)

        print("\n".join(f'{key:<16}{value:>10.4f}' for key, value in metrics.items()))
