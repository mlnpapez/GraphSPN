import os
import json
import pandas as pd
import graphspn

from tqdm import tqdm
from pylatex import Document, Package, NoEscape, TikZOptions, TikZ, TikZNode

IGNORE = [
    'device',
    'seed',
    'nll_trn_approx',
    'nll_val_approx',
    'nll_tst_approx',
    'num_parameters',
    'file_path',
    ]

def baseline_models_qm9():
    column_working_names = ['model', 'validity', 'uniqueness', 'novelty']

    data = {'model':      ['GVAE', 'GraphNVP', 'GRF', 'GraphAF', 'GraphDF', 'MoFlow', 'ModFlow'],
            'validity':   [  60.2,       83.1,  84.5,      67.0,      82.7,     89.0,      99.1],
            'uniqueness': [   9.3,       99.2,  66.0,      94.2,      97.6,     98.5,      99.3],
            'novelty':    [  80.9,       58.2,  58.6,      88.8,      98.1,     96.4,     100.0]}

    df = pd.DataFrame.from_dict(data)

    print(df)


def texable(x):
    return [n.replace('_', '-') for n in x]

def latexify_style(df, path, row_names=None, column_names=None, precision=2):
    if row_names is not None:
        df.replace(row_names, inplace=True)
    if column_names is not None:
        df.rename(columns={o: n for o, n in zip(df.columns, column_names)}, inplace=True)

    s = df.style.highlight_min(subset=df.columns[1:], axis=1, props='color:{red};' 'textbf:--rwrap;')
    s.hide()
    # s.relabel_index(texable(df.columns), axis=1)
    # formatter_a = {'dataset': lambda x: x.replace('_', '-')}
    # formatter_b = {c: '{:3.2f}' for c in df.columns if c != 'dataset'}
    # s.format({**formatter_a, **formatter_b})
    s.format(precision=precision)
    # s.caption = 'The test negative log-likelihood.'
    # s.to_latex(path, hrules=True, column_format='lm{1.8cm}rm{1.4cm}rm{1.4cm}rm{1.4cm}rm{2.2cm}rm{2.2cm}')
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
    column_working_names = ['dataset'] + list(names_models)
    column_display_names = ['Dataset'] + [m.__name__ for m in graphspn.MODELS.values()]

    file_unsu = bestmodels_dir + f'tabular_nll_best_models'
    frame_unsu = pd.DataFrame(0., index=range(len(names_datasets)), columns=column_working_names)

    pth_datasets = {}
    for i, dataset in tqdm(enumerate(names_datasets), leave=False):
        frame_unsu.loc[i, 'dataset'] = dataset
        pth_models = {}
        for model in tqdm(names_models, leave=False):
            res_frames = []
            path = evaluation_dir + dataset + '/' + model + '/'
            for f in os.listdir(path):
                data = pd.read_csv(path + f)
                data['dataset'] = dataset
                data['file_path'] = f.replace('.csv', '.pt')
                res_frames.append(data)

            cat_frames = pd.concat(res_frames)
            grp_frames = cat_frames.groupby(list(filter(lambda x: x not in IGNORE, cat_frames.columns)))
            agg_frames = grp_frames.agg({
                'nll_trn_approx': 'mean',
                'nll_val_approx': 'mean',
                'nll_tst_approx': 'mean',
                'file_path': 'first'
            })
            best_value = agg_frames.nsmallest(n=1, columns='nll_val_approx')
            best_frame = grp_frames.get_group(agg_frames['nll_val_approx'].idxmin())

            frame_unsu.loc[i, model] = best_value['nll_tst_approx'][0]
            pth_models[model] = list(best_frame['file_path'])

        pth_datasets[dataset] = pth_models

    with open(f'{file_unsu}.json', 'w') as f:
        json.dump(pth_datasets, f, indent=4)

    latexify_style(frame_unsu, f'{file_unsu}.tab', row_names=datasets.DEBD_NAMES, column_names=column_display_names)
    latexify_table(f'{file_unsu}.tab', f'{file_unsu}')


if __name__ == "__main__":
    checkpoint_dir = 'results/tabular/model_checkpoint/'
    evaluation_dir = 'results/tabular/model_evaluation/'
    hyperparam_dir = 'results/tabular/model_outputlogs/'
    bestmodels_dir = 'results/tabular/'

    names_models   = graphspn.MODELS.keys()

    baseline_models_qm9()

    # find_best_models(names_datasets, names_models, evaluation_dir, bestmodels_dir)
    # evaluate_metrics('samp', names_datasets, names_models, checkpoint_dir, bestmodels_dir)

    # doc = Document(documentclass='standalone', document_options=('preview'), geometry_options={'margin': '0cm'})
    # doc.append(NoEscape(r'\usetikzlibrary{positioning}'))

    # with doc.create(TikZ(options=TikZOptions({'node distance': '4px and 4px'}))) as pic:
    #     pic.append(TikZNode(
    #         text=f'\includegraphics{{results/tabular/join_metr.pdf}}',
    #         options=TikZOptions({'label': '{[yshift=-7px]{' + f'NLL (nats)' '}}'}),
    #         handle=f'join_metr'))
    #     pic.append(TikZNode(
    #         text=f'\includegraphics{{results/tabular/join_time.pdf}}',
    #         options=TikZOptions({'below': f'of join_metr','label': '{[yshift=-7px]{' + f'Time (sec)' '}}'}),
    #         handle=f'join_time'))

    #     pic.append(TikZNode(
    #         text=f'\includegraphics{{results/tabular/marg_metr_0.2.pdf}}',
    #         options=TikZOptions({'right': f'of join_metr', 'label': '{[yshift=-7px]{' + 'NLL (nats), Marginalization rate: 0.2' '}}'}),
    #         handle=f'marg_metr'))
    #     pic.append(TikZNode(
    #         text=f'\includegraphics{{results/tabular/marg_time_0.2.pdf}}',
    #         options=TikZOptions({'below': f'of marg_metr', 'label': '{[yshift=-7px]{' + f'Time (sec)' '}}'}),
    #         handle=f'marg_time'))

    #     pic.append(TikZNode(
    #         text=f'\includegraphics{{results/tabular/samp_metr_1000.pdf}}',
    #         options=TikZOptions({'right': f'of marg_metr', 'label': '{[yshift=-7px]{' + f'MSE (-)' '}}'}),
    #         handle=f'samp_metr'))
    #     pic.append(TikZNode(
    #         text=f'\includegraphics{{results/tabular/samp_time_1000.pdf}}',
    #         options=TikZOptions({'below': f'of samp_metr', 'label': '{[yshift=-7px]{' + f'Time (sec)' '}}'}),
    #         handle=f'samp_time'))

    #     pic.append(TikZNode(
    #         text=f'\includegraphics{{results/tabular/nove_metr.pdf}}',
    #         options=TikZOptions({'right': f'of samp_metr', 'label': '{[yshift=-7px]{' + f'Novelty (-)' '}}'}),
    #         handle=f'nove_metr'))

    #     pic.append(TikZNode(
    #         text=f'\includegraphics{{results/tabular/num_parameters.pdf}}',
    #         options=TikZOptions({'below': f'of join_time', 'label': '{[yshift=-7px]{' + f'Number of parameters (-)' '}}'}),
    #         handle=f'num_parameters'))

    # doc.generate_pdf('evaluate_unsupervised_all', clean_tex=True)
