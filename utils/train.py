import os
import torch
import torch.optim as optim
import pandas as pd

from tqdm import tqdm
from rdkit.Chem.Draw import MolsToGridImage

from utils.evaluate import evaluate_molecules, resample_invalid_mols, count_parameters

IGNORED_HYPERPARS = [
    'atom_list',
    'mask_row_stride_list',
    'mask_row_size_list',
    'optimizer'
]


def flatten_dict(d, input_key=''):
    if isinstance(d, dict):
        return {k if input_key else k: v for key, value in d.items() for k, v in flatten_dict(value, key).items()}
    else:
        return {input_key: d}

def dict2str(d):
    return '_'.join([f'{key}={value}' for key, value in d.items() if key not in IGNORED_HYPERPARS])



def run_epoch(model, loader, optimizer=[], verbose=False):
    nll_sum = 0.
    for b in tqdm(loader, leave=False, disable=verbose):
        x = b['x'].to(model.device) # (256,9,5)
        a = b['a'].to(model.device) # (256,4,9,9)
        nll = -model.logpdf(x, a)
        nll_sum += nll
        if optimizer:
            optimizer.zero_grad()
            nll.backward()
            optimizer.step()

    return nll_sum.item()

METRIC_TYPES = ['valid', 'unique', 'novel', 'score']

def train(
        model,
        loader_trn,
        loader_val,
        smiles_trn,
        hyperpars,
        checkpoint_dir,
        num_nonimproving_epochs=200,
        verbose=False,
        metric_type='ll_val'
    ):
    optimizer = optim.Adam(model.parameters(), **hyperpars['optimizer_hyperpars'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)

    lookahead_counter = num_nonimproving_epochs
    if metric_type in METRIC_TYPES:
        best_metric = 0e0
    else:
        best_metric = 1e6
    best_model_path = None
    save_model = False

    for epoch in range(hyperpars['num_epochs']):
        model.train()
        nll_trn = run_epoch(model, loader_trn, verbose=verbose, optimizer=optimizer)
        scheduler.step()
        model.eval()

        x_sam, a_sam = model.sample(2000)
        metrics = evaluate_molecules(x_sam, a_sam, smiles_trn, hyperpars['atom_list'], metrics_only=True)
        metrics_str = f'v={metrics["valid"]:.2f}, u={metrics["unique"]:.2f}, n={metrics["novel"]:.2f}, s={metrics["score"]:.2f}'

        if metric_type in METRIC_TYPES:
            metric = metrics[metric_type]
            print(f'epoch {epoch:3d}: ll_trn={-nll_trn:.4f}, ' + metrics_str)

            if metric > best_metric:
                best_metric = metric
                lookahead_counter = num_nonimproving_epochs
                save_model = True
            else:
                lookahead_counter -= 1
        else:
            metric = run_epoch(model, loader_val, verbose=verbose)
            print(f'epoch {epoch:3d}: ll_trn={-nll_trn:.4f}, ll_val={-metric:.4f}, ' + metrics_str)

            if metric < best_metric:
                best_metric = metric
                lookahead_counter = num_nonimproving_epochs
                save_model = True
            else:
                lookahead_counter -= 1

        if lookahead_counter == 0:
            break

        if save_model == True:
            dir = checkpoint_dir + f'{hyperpars["dataset"]}/{hyperpars["model"]}/'

            if os.path.isdir(dir) != True:
                os.makedirs(dir)
            if best_model_path != None:
                os.remove(best_model_path)
            path = dir + dict2str(flatten_dict(hyperpars)) + '.pt'
            torch.save(model, path)
            best_model_path = path
            save_model == False

    return best_model_path

def evaluate(
        model,
        loader_trn,
        loader_val,
        loader_tst,
        smiles_trn,
        hyperpars,
        evaluation_dir,
        num_samples=4000,
        compute_nll=True,
        canonical=True
    ):
    model.eval()

    x_sam, a_sam = model.sample(num_samples)
    x_res, a_res = resample_invalid_mols(model, num_samples, hyperpars['atom_list'], hyperpars['max_atoms'], canonical)

    mols_res_f, _, metrics_res_f = evaluate_molecules(x_sam, a_sam, smiles_trn, hyperpars['atom_list'], correct_mols=False, affix='res_f_', canonical=canonical)
    mols_res_t, _, metrics_res_t = evaluate_molecules(x_res, a_res, smiles_trn, hyperpars['atom_list'], correct_mols=False, affix='res_t_', canonical=canonical)
    mols_cor_t, _, metrics_cor_t = evaluate_molecules(x_sam, a_sam, smiles_trn, hyperpars['atom_list'], correct_mols=True,  affix='cor_t_', canonical=canonical)

    if compute_nll == True:
        nll_trn_approx = run_epoch(model, loader_trn)
        nll_val_approx = run_epoch(model, loader_val)
        nll_tst_approx = run_epoch(model, loader_tst)
        metrics_neglogliks = {
            'nll_trn_approx': nll_trn_approx,
            'nll_val_approx': nll_val_approx,
            'nll_tst_approx': nll_tst_approx
        }
    else:
        metrics_neglogliks = {}

    metrics = {**metrics_res_f,
               **metrics_res_t,
               **metrics_cor_t,
               **metrics_neglogliks,
               "num_params": count_parameters(model)}

    dir = evaluation_dir + f'metrics/{hyperpars["dataset"]}/{hyperpars["model"]}/'
    if os.path.isdir(dir) != True:
        os.makedirs(dir)
    path = dir + dict2str(flatten_dict(hyperpars))
    df = pd.DataFrame.from_dict({**flatten_dict(hyperpars), **metrics}, 'index').transpose()
    df.to_csv(path + '.csv', index=False)

    dir = evaluation_dir + f'images/{hyperpars["dataset"]}/{hyperpars["model"]}/'
    if os.path.isdir(dir) != True:
        os.makedirs(dir)
    path = dir + dict2str(flatten_dict(hyperpars))

    img_res_f = MolsToGridImage(mols=mols_res_f[0:64], molsPerRow=8, subImgSize=(200, 200), useSVG=False)
    img_res_t = MolsToGridImage(mols=mols_res_t[0:64], molsPerRow=8, subImgSize=(200, 200), useSVG=False)
    img_cor_t = MolsToGridImage(mols=mols_cor_t[0:64], molsPerRow=8, subImgSize=(200, 200), useSVG=False)

    img_res_f.save(path + f'_img_res_f.png')
    img_res_t.save(path + f'_img_res_t.png')
    img_cor_t.save(path + f'_img_cor_t.png')

    return metrics
