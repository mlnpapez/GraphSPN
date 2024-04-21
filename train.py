import os
import json
import torch
import torch.optim as optim
import pandas as pd
import utils

from graphspn import *
from datasets import MolecularDataset, load_dataset
from rdkit.Chem.Draw import MolsToGridImage
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
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
 
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


def evaluate(model, loader_trn, loader_val, loader_tst, smiles_trn, hyperpars, evaluation_dir, num_samples=1000):
    model.eval()

    # nll_trn_approx = run_epoch(model, loader_trn)
    nll_val_approx = run_epoch(model, loader_val)
    nll_tst_approx = run_epoch(model, loader_tst)

    molecules_sam, smiles_sam = model.sample(num_samples)
    molecules_res, smiles_res = resample_invalid_mols(model, num_samples)

    molecules_res_f, _, metrics_resample_f = utils.evaluate_molecules(molecules_sam, smiles_sam, smiles_trn, num_samples, correct_mols=False, affix='res_f_')
    molecules_res_t, _, metrics_resample_t = utils.evaluate_molecules(molecules_res, smiles_res, smiles_trn, num_samples, correct_mols=False, affix='res_t_')
    molecules_cor_t, _, metrics_correction = utils.evaluate_molecules(molecules_sam, smiles_res, smiles_trn, num_samples, correct_mols=True,  affix='cor_t_')
    metrics_neglogliks = {
        # 'nll_trn_approx': nll_trn_approx,
        'nll_val_approx': nll_val_approx,
        'nll_tst_approx': nll_tst_approx
    }
    metrics = {**metrics_resample_f, **metrics_resample_t, **metrics_correction, **metrics_neglogliks}

    dir = evaluation_dir + 'metrics/' + hyperpars['model'] + '/'
    if os.path.isdir(dir) != True:
        os.makedirs(dir)
    path = dir + dict2str(flatten_dict(hyperpars))
    df = pd.DataFrame.from_dict({**flatten_dict(hyperpars), **metrics}, 'index').transpose()
    df.to_csv(path + '.csv', index=False)

    dir = evaluation_dir + 'images/' + hyperpars['model'] + '/'
    if os.path.isdir(dir) != True:
        os.makedirs(dir)
    path = dir + dict2str(flatten_dict(hyperpars))

    img_res_f = MolsToGridImage(mols=molecules_res_f[0:64], molsPerRow=8, subImgSize=(200, 200), useSVG=False)
    img_res_t = MolsToGridImage(mols=molecules_res_t[0:64], molsPerRow=8, subImgSize=(200, 200), useSVG=False)
    img_cor_t = MolsToGridImage(mols=molecules_cor_t[0:64], molsPerRow=8, subImgSize=(200, 200), useSVG=False)

    img_res_f.save(path + f'_img_res_f.png')
    img_res_t.save(path + f'_img_res_t.png')
    img_cor_t.save(path + f'_img_cor_t.png')

    return metrics


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')

    dataset = 'zinc250k'
    name = 'graphspn_naive_cat_a'

    checkpoint_dir = 'results/training/model_checkpoint/'
    trainepoch_dir = 'results/training/model_trainepoch/'
    evaluation_dir = 'results/training/model_evaluation/'
    hyperparam_dir = 'configs/training/model_hyperparam/'

    with open('config/' + f'{name}.json', 'r') as f:
        hyperpars = json.load(f)

    model = MODELS[name](**hyperpars['model_hyperpars'])

    if 'deq' in name:
        loader_trn, loader_val, loader_tst = load_dataset(dataset, hyperpars['batch_size'], ohe=True)
    else:
        loader_trn, loader_val, loader_tst = load_dataset(dataset, hyperpars['batch_size'], ohe=False)

    x_trn, _, _ = load_dataset(dataset, 0, raw=True)
    smiles_trn = [x['s'] for x in x_trn]

    path = train(model, loader_trn, loader_val, hyperpars, checkpoint_dir, trainepoch_dir)
    model = torch.load(path)
    metrics = evaluate(model, loader_trn, loader_val, loader_tst, smiles_trn, hyperpars, evaluation_dir)

    print("\n".join(f'{key:<16}{value:>10.4f}' for key, value in metrics.items()))
