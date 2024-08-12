import os
from numpy import NaN
import pandas as pd

from tqdm import tqdm
from pylatex import Document, Package, NoEscape

MODEL_NAMES_TABLE = {
    'graphspn_zero_none': 'GraphSPN: None',
    'graphspn_zero_full': 'GraphSPN: Full',
    'graphspn_zero_rand': 'GraphSPN: Rand',
    'graphspn_zero_sort': 'GraphSPN: Sort',
    'graphspn_zero_kary': 'GraphSPN: $k$-ary',
    'graphspn_zero_free': 'GraphSPN: Ind',
}

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

COLUMN_WORKING_NAMES = ['model', 'validity', 'validitywocheck',    'uniqueness', 'novelty']
COLUMN_DISPLAY_NAMES = ['Model', 'Validity', 'Validity w/o check', 'Uniqueness', 'Novelty']

def baseline_models_qm9():
    data = [['GraphVAE', 'GVAE', 'CVAE', 'RVAE', 'GraphNVP', 'GRF', 'GraphAF', 'GraphDF', 'MoFlow', 'ModFlow'],
            [      55.7,   60.2,   10.2,   96.6,       83.1,  84.5,     100.0,     100.0,    100.0,       NaN],
            [       NaN,    NaN,    NaN,    NaN,        NaN,   NaN,      67.0,      82.7,     89.0,      99.1],
            [      76.0,    9.3,   67.5,    NaN,       99.2,  66.0,      94.2,      97.6,     98.5,      99.3],
            [      61.6,   80.9,   90.0,   95.5,       58.2,  58.6,      88.8,      98.1,     96.4,     100.0]]
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


def find_best_models(dataset, names_models, evaluation_dir):
    data_unsu = pd.DataFrame(0., index=range(len(names_models)), columns=COLUMN_WORKING_NAMES)

    for i, model in tqdm(enumerate(names_models), leave=False):
        res_frames = []
        path = evaluation_dir + f'{model}/'
        for f in os.listdir(path):
            data = pd.read_csv(path + f)
            data['file_path'] = f.replace('.csv', '.pt')
            res_frames.append(data)

        cat_frames = pd.concat(res_frames)
        grp_frames = cat_frames.groupby(list(filter(lambda x: x not in IGNORE, cat_frames.columns)))
        agg_frames = grp_frames.agg({
            'res_f_valid': 'mean',
            'file_path': 'first'
        })
        best_frame = grp_frames.get_group(agg_frames['res_f_valid'].idxmax())

        data_unsu.loc[i] = [MODEL_NAMES_TABLE[model], 100*best_frame['cor_t_valid'].mean(), 100*best_frame['res_f_valid'].mean(), 100*best_frame['res_f_unique'].mean(), 100*best_frame['res_f_novel'].mean()]

    return data_unsu


if __name__ == "__main__":
    # checkpoint_dir = 'results/gridsearch/model_checkpoint/'
    evaluation_dir = 'results/gridsearch/model_evaluation/metrics/qm9/'
    # hyperparam_dir = 'results/gridsearch/model_outputlogs/'
    # bestmodels_dir = 'results/gridsearch/'

    # models   = [name for name in graphspn.MODELS.keys() if name not in ['graphspn_naive_deq_b', 'graphspn_naive_deq_h']]
    models = os.listdir(evaluation_dir)
    # models = ['graphspn_zero_none',
    #           'graphspn_zero_rand',
    #           'graphspn_zero_sort',
    #           'graphspn_zero_kary',
    #           'graphspn_zero_free']

    baselines = baseline_models_qm9()
    
    ourmodels = find_best_models('qm9', models, evaluation_dir)
    # baselines['score'] = baselines.apply(lambda row: row['validity']*row['uniqueness']*row['novelty']/10000, axis=1)

    allmodels = pd.concat([baselines, ourmodels], ignore_index=True)

    latexify_style(allmodels, 'qm9.tab', column_names=COLUMN_DISPLAY_NAMES)
    latexify_table('qm9.tab', 'qm9')
