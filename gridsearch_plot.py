import os
import pandas as pd

from pylatex import Document, TikZ, NoEscape

LEGENDS = {
    'graphspn_marg_none': 'mnone',
    'graphspn_marg_full': 'mfull',
    'graphspn_marg_rand': 'mrand',
    'graphspn_marg_sort': 'msort',
    'graphspn_marg_kary': 'mkary',
    'graphspn_zero_rand': 'zrand',
}

# https://tikz.dev/pgfplots/reference-markers
MARKS = {
    'graphspn_marg_none': 'x',
    'graphspn_marg_full': '*',
    'graphspn_marg_rand': 'square',
    'graphspn_marg_sort': 'triangle',
    'graphspn_marg_kary': 'star',
    'graphspn_zero_rand': 'asterisk',
}

def nextgrouplot(models, ydata, ylabel, evaluation_dir):
    pic.append(NoEscape(f'\\nextgroupplot[xlabel={{number of parameters (-)}}, ylabel={{{ylabel} (-)}}]'))
    for i, m in enumerate(models):
        df = pd.concat([pd.read_csv(evaluation_dir + m + '/' + f) for f in os.listdir(evaluation_dir + m)])
        coordinates = list(df[['num_params', ydata]].itertuples(index=False, name=None))
        pic.append(NoEscape(f'\\addplot [color=c{i}, mark={MARKS[m]}, only marks] coordinates {{' + ' '.join(str(x) for x in coordinates) + '};' + f'\\addlegendentry{{{LEGENDS[m]}}};'))


if __name__ == "__main__":
    evaluation_dir = 'results/training/model_evaluation/metrics/qm9/'

    models = os.listdir(evaluation_dir)

    doc = Document(documentclass='standalone', document_options=('preview'), geometry_options={'margin': '0cm'})
    doc.packages.append(NoEscape(r'\usepackage{pgfplots}'))
    doc.packages.append(NoEscape(r'\usepgfplotslibrary{groupplots}'))

    doc.packages.append(NoEscape(r'\definecolor{c0}{RGB}{27,158,119}'))
    doc.packages.append(NoEscape(r'\definecolor{c1}{RGB}{117,112,179}'))
    doc.packages.append(NoEscape(r'\definecolor{c2}{RGB}{217,95,2}'))
    doc.packages.append(NoEscape(r'\definecolor{c3}{RGB}{231,41,138}'))
    doc.packages.append(NoEscape(r'\definecolor{c4}{RGB}{230,171,2}'))

    with doc.create(TikZ()) as pic:
        pic.append(NoEscape(r'\begin{groupplot}[group style={group size=4 by 3, horizontal sep=35pt, vertical sep=35pt},height=5cm,width=6.4cm,xmode=log,ymin=0,ymax=1,legend style={font=\tiny,fill=none,draw=none},legend pos=north west,label style={font=\small},y label style={at={(0.08,0.5)}}]'))

        nextgrouplot(models, 'res_f_valid',  'validity',   evaluation_dir)
        nextgrouplot(models, 'res_f_unique', 'uniqueness', evaluation_dir)
        nextgrouplot(models, 'res_f_novel',  'novelty',    evaluation_dir)
        nextgrouplot(models, 'res_f_score',  'score',      evaluation_dir)

        nextgrouplot(models, 'res_t_valid',  'validity',   evaluation_dir)
        nextgrouplot(models, 'res_t_unique', 'uniqueness', evaluation_dir)
        nextgrouplot(models, 'res_t_novel',  'novelty',    evaluation_dir)
        nextgrouplot(models, 'res_t_score',  'score',      evaluation_dir)

        nextgrouplot(models, 'cor_t_valid',  'validity',   evaluation_dir)
        nextgrouplot(models, 'cor_t_unique', 'uniqueness', evaluation_dir)
        nextgrouplot(models, 'cor_t_novel',  'novelty',    evaluation_dir)
        nextgrouplot(models, 'cor_t_score',  'score',      evaluation_dir)

        pic.append(NoEscape(r'\end{groupplot}'))
    doc.generate_pdf('train', clean_tex=False)
