import os
import json
import torch
import torch.optim as optim
import pandas as pd

from graphspn import GraphSPN
from preprocess import MolecularDataset, load_qm9
from tqdm import tqdm


def flatten_dict(d, input_key=''):
    if isinstance(d, dict):
        return {k if input_key else k: v for key, value in d.items() for k, v in flatten_dict(value, key).items()}
    else:
        return {input_key: d}

def dict2str(d):
    return '_'.join([f'{key}={value}' for key, value in d.items()])

def run_epoch(model, loader, optimizer=[]):
    nll_sum = 0.
    for x in tqdm(loader, leave=False):
        nll = -model.logpdf(x)
        nll_sum += nll * len(x)
        if optimizer:
            optimizer.zero_grad()
            nll.backward()
            optimizer.step()

    return (nll_sum / len(loader.dataset)).item()

def train(model, loader_trn, loader_val, hyperpars, checkpoint_dir, trainepoch_dir, num_nonimproving_epochs=30):
    optimizer = optim.Adam(model.parameters(), **hyperpars['optimizer_hyperpars'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=45, gamma=0.1)
 
    lookahead_counter = num_nonimproving_epochs
    best_nll_val = 1e6
    best_model_path = None

    for epoch in range(hyperpars['num_epochs']):
        model.train()
        nll_trn = run_epoch(model, loader_trn, optimizer)
        scheduler.step()

        model.eval()
        nll_val = run_epoch(model, loader_val)

        dir = trainepoch_dir + 'samples/' + hyperpars['model'] + '/'
        if os.path.isdir(dir) != True:
            os.makedirs(dir)

        print(f'epoch {epoch:3d}: ll_trn={-nll_trn:.4f}, ll_val={-nll_val:.4f}')

        if nll_val < best_nll_val:
            best_nll_val = nll_val
            lookahead_counter = num_nonimproving_epochs

            dir = checkpoint_dir + hyperpars['model'] + '/'

            if os.path.isdir(dir) != True:
                os.makedirs(dir)
            if best_model_path != None:
                os.remove(best_model_path)
            path = dir + dict2str(flatten_dict(hyperpars)) + '.pt'
            torch.save(model, path)
            best_model_path = path
        else:
            lookahead_counter -= 1

        if lookahead_counter == 0:
            break

    return best_model_path


def evaluate(model, loader_trn, loader_val, loader_tst, hyperpars, evaluation_dir):
    model.eval()

    nll_trn_approx = run_epoch(model, loader_trn)
    nll_val_approx = run_epoch(model, loader_val)
    nll_tst_approx = run_epoch(model, loader_tst)

    metrics = {
        'nll_trn_approx': nll_trn_approx,
        'nll_val_approx': nll_val_approx,
        'nll_tst_approx': nll_tst_approx
    }

    dir = evaluation_dir + hyperpars['model'] + '/'

    df = pd.DataFrame.from_dict({**flatten_dict(hyperpars), **metrics}, 'index').transpose()
    if os.path.isdir(dir) != True:
        os.makedirs(dir)
    path = dir + dict2str(flatten_dict(hyperpars)) + '.csv'
    df.to_csv(path, index=False)

    return metrics


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')

    name = 'graphspn'

    checkpoint_dir = 'results/training/model_checkpoint/'
    trainepoch_dir = 'results/training/model_trainepoch/'
    evaluation_dir = 'results/training/model_evaluation/'
    # hyperparam_dir = 'configs/training/model_hyperparam/'

    with open('config/' + f'{name}.json', 'r') as fp:
        hyperpars = json.load(fp)

    model = GraphSPN(**hyperpars['model_hyperpars'])

    loader_trn, loader_val, loader_tst = load_qm9(hyperpars['batch_size'])

    path = train(model, loader_trn, loader_val, hyperpars, checkpoint_dir, trainepoch_dir)
    model = torch.load(path)
    metrics = evaluate(model, loader_trn, loader_val, loader_tst, hyperpars, evaluation_dir)

    print("\n".join(f'{key:<20}{value:>10.4f}' for key, value in metrics.items()))
