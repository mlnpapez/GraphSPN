import os
import pandas as pd
import graphspn

from tqdm import tqdm
from pylatex import Document, Package, NoEscape

IGNORE = [
    'atom_list',
    'device',
    'seed',
    'res_f_valid',
    'res_t_valid',
    'cor_t_valid',
    'res_f_unique',
    'cor_t_unique',
    'res_t_unique',
    'res_f_novel',
    'res_t_novel',
    'cor_t_novel',
    'res_f_novel_abs',
    'res_t_novel_abs',
    'cor_t_novel_abs',
    'res_f_score',
    'res_t_score',
    'cor_t_score',
    'res_f_unique_abs',
    'res_t_unique_abs',
    'cor_t_unique_abs',
    'nll_val_approx',
    'nll_tst_approx',
    'num_params',
    'file_path',
    ]

COLUMN_WORKING_NAMES = ['model', 'validity', 'uniqueness', 'novelty', 'score']
COLUMN_DISPLAY_NAMES = ['Model', 'Validity', 'Uniqueness', 'Novelty', 'Score']

def baseline_models_qm9():
    data = [['GVAE', 'GraphNVP', 'GRF', 'GraphAF', 'GraphDF', 'MoFlow', 'ModFlow'],
            [  60.2,       83.1,  84.5,      67.0,      82.7,     89.0,      99.1],
            [   9.3,       99.2,  66.0,      94.2,      97.6,     98.5,      99.3],
            [  80.9,       58.2,  58.6,      88.8,      98.1,     96.4,     100.0]]
    return pd.DataFrame.from_dict({k:v for k, v, in zip(COLUMN_WORKING_NAMES, data)})


def texable(x):
    return [n.replace('_', '-') for n in x]

def latexify_style(df, path, row_names=None, column_names=None, precision=2):
    if row_names is not None:
        df.replace(row_names, inplace=True)
    if column_names is not None:
        df.rename(columns={o: n for o, n in zip(df.columns, column_names)}, inplace=True)

    s = df.style.highlight_max(subset=df.columns[1:], axis=0, props='color:{red};' 'textbf:--rwrap;')
    s.hide()
    s.format(precision=precision)
    # s.caption = 'Results.'
    s.to_latex(path, hrules=True)

def latexify_table(r_name, w_name, clean_tex=True):
    with open(r_name) as f:
        table = f.read()

    doc = Document(documentclass='standalone', document_options=('preview'), geometry_options={'margin': '0cm'})
    doc.packages.append(Package('booktabs'))
    doc.packages.append(Package('xcolor', options='table'))
    doc.append(NoEscape(table))
    doc.generate_pdf(f'{w_name}', clean_tex=clean_tex)


def find_best_models(dataset, names_models, evaluation_dir, bestmodels_dir):
    data_unsu = pd.DataFrame(0., index=range(len(names_models)), columns=COLUMN_WORKING_NAMES)

    for i, model in tqdm(enumerate(names_models), leave=False):
        res_frames = []
        path = evaluation_dir + 'metrics/' + f'{dataset}/' + f'{model}/'
        for f in os.listdir(path):
            data = pd.read_csv(path + f)
            data['file_path'] = f.replace('.csv', '.pt')
            res_frames.append(data)

        cat_frames = pd.concat(res_frames)
        grp_frames = cat_frames.groupby(list(filter(lambda x: x not in IGNORE, cat_frames.columns)))
        agg_frames = grp_frames.agg({
            'res_f_score': 'mean',
            'file_path': 'first'
        })
        best_frame = grp_frames.get_group(agg_frames['res_f_score'].idxmax())

        data_unsu.loc[i] = [model.replace('_', '-'), 100*best_frame['res_f_valid'].mean(), 100*best_frame['res_f_unique'].mean(), 100*best_frame['res_f_novel'].mean(), 100*best_frame['res_f_score'].mean()]

    return data_unsu


if __name__ == "__main__":
    checkpoint_dir = 'results/gridsearch/model_checkpoint/'
    evaluation_dir = 'results/gridsearch/model_evaluation/'
    hyperparam_dir = 'results/gridsearch/model_outputlogs/'
    bestmodels_dir = 'results/gridsearch/'

    names_models   = [name for name in graphspn.MODELS.keys() if name not in ['graphspn_naive_deq_b', 'graphspn_naive_deq_h']]

    baselines = baseline_models_qm9()
    
    ourmodels = find_best_models('qm9', names_models, evaluation_dir, bestmodels_dir)
    baselines['score'] = baselines.apply(lambda row: row['validity']*row['uniqueness']*row['novelty']/10000, axis=1)

    allmodels = pd.concat([baselines, ourmodels], ignore_index=True)

    latexify_style(allmodels, 'qm9.tab', column_names=COLUMN_DISPLAY_NAMES + ['Score'])
    latexify_table('qm9.tab', 'qm9')
