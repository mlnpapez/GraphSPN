import json
import torch
import utils
import datasets
import graphspn
import graphspn_zero
import graphspn_marg

from rdkit import RDLogger


MODELS = {**graphspn.MODELS, **graphspn_zero.MODELS, **graphspn_marg.MODELS}


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    RDLogger.DisableLog('rdApp.*')

    dataset = 'qm9'
    models = ['graphspn_marg_rand'] # MODELS.keys()

    checkpoint_dir = 'results/training/model_checkpoint/'
    trainepoch_dir = 'results/training/model_trainepoch/'
    evaluation_dir = 'results/training/model_evaluation/'
    hyperparam_dir = 'configs/training/model_hyperparam/'

    for name in models:
        with open(f'config/{dataset}/{name}.json', 'r') as f:
            hyperpars = json.load(f)

        model = MODELS[name](**hyperpars['model_hyperpars'])
        print(model)

        if 'deq' in name:
            loader_trn, loader_val, loader_tst = datasets.load_dataset(hyperpars['dataset'], hyperpars['batch_size'], ohe=True)
        else:
            loader_trn, loader_val, loader_tst = datasets.load_dataset(hyperpars['dataset'], hyperpars['batch_size'], ohe=False)

        x_trn, _, _ = datasets.load_dataset(hyperpars['dataset'], 0, raw=True)
        smiles_trn = [x['s'] for x in x_trn]

        path = utils.train(model, loader_trn, loader_val, smiles_trn, hyperpars, checkpoint_dir, trainepoch_dir)
        model = torch.load(path)
        metrics = utils.evaluate(model, loader_trn, loader_val, loader_tst, smiles_trn, hyperpars, evaluation_dir, compute_nll=False)

        print("\n".join(f'{key:<16}{value:>10.4f}' for key, value in metrics.items()))
