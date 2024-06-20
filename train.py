import json
import torch
import utils
import datasets
import graphspn_naive
import graphspn_zero
import graphspn_marg
import graphspn_back

from rdkit import RDLogger


MODELS = {**graphspn_naive.MODELS, **graphspn_zero.MODELS, **graphspn_marg.MODELS, **graphspn_back.MODELS}


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    RDLogger.DisableLog('rdApp.*')

    dataset = 'qm9'
    models = ['graphspn_back_none'] # MODELS.keys()

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

        # if 'sort' in name:
        #     loader_trn, smiles_trn = datasets.permute_dataset(loader_trn, dataset, permutation='canonical')
        # else:
        #     loader_trn, smiles_trn = datasets.permute_dataset(loader_trn, dataset)
        loader_trn, smiles_trn = datasets.permute_dataset(loader_trn, dataset, permutation='canonical')

        path = utils.train(model, loader_trn, loader_val, smiles_trn, hyperpars, checkpoint_dir, trainepoch_dir)
        model = torch.load(path)
        metrics = utils.evaluate(model, loader_trn, loader_val, loader_tst, smiles_trn, hyperpars, evaluation_dir, compute_nll=False)

        print("\n".join(f'{key:<16}{value:>10.4f}' for key, value in metrics.items()))
