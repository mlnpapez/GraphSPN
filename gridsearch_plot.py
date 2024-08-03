import os
import pandas as pd

from pylatex import Document, TikZ, NoEscape

LEGENDS = {
    'graphspn_zero_none': 'znone',
    'graphspn_zero_full': 'zfull',
    'graphspn_zero_rand': 'zrand',
    'graphspn_zero_sort': 'zsort',
    'graphspn_zero_kary': 'zkary',
    'graphspn_zero_free': 'zfree',
}

# https://tikz.dev/pgfplots/reference-markers
MARKS = {
    'graphspn_zero_none': '+',
    'graphspn_zero_full': '*',
    'graphspn_zero_rand': 'o',
    'graphspn_zero_sort': 'halfcircle',
    'graphspn_zero_kary': 'diamond',
    'graphspn_zero_free': 'pentagon',
}

def nextgrouplot(models, ydata, ylabel, evaluation_dir):
    pic.append(NoEscape(f'\\nextgroupplot[xlabel={{Number of parameters (-)}}, ylabel={{{ylabel} (-)}}]'))
    for i, m in enumerate(models):
        df = pd.concat([pd.read_csv(evaluation_dir + m + '/' + f) for f in os.listdir(evaluation_dir + m)])
        coordinates = list(df[['num_params', ydata]].itertuples(index=False, name=None))
        pic.append(NoEscape(f'\\addplot [color=c{i}, mark={MARKS[m]}, only marks] coordinates {{' + ' '.join(str(x) for x in coordinates) + '};' + f'\\addlegendentry{{{LEGENDS[m]}}};'))


if __name__ == "__main__":
    evaluation_dir = 'results/linesearch/model_evaluation/metrics/qm9/'

    models = os.listdir(evaluation_dir)
    # models = ['graphspn_zero_none', 'graphspn_zero_rand', 'graphspn_zero_sort', 'graphspn_zero_kary', 'graphspn_zero_free']

    doc = Document(documentclass='standalone', document_options=('preview'), geometry_options={'margin': '1cm'})
    doc.packages.append(NoEscape(r'\usepackage{pgfplots}'))
    doc.packages.append(NoEscape(r'\usepgfplotslibrary{groupplots}'))

    doc.packages.append(NoEscape(r'\definecolor{c0}{RGB}{27,158,119}'))
    doc.packages.append(NoEscape(r'\definecolor{c1}{RGB}{117,112,179}'))
    doc.packages.append(NoEscape(r'\definecolor{c2}{RGB}{217,95,2}'))
    doc.packages.append(NoEscape(r'\definecolor{c3}{RGB}{231,41,138}'))
    doc.packages.append(NoEscape(r'\definecolor{c4}{RGB}{230,171,2}'))
    doc.packages.append(NoEscape(r'\definecolor{c5}{RGB}{166,118,29}'))

    doc.packages.append(NoEscape(r'\definecolor{c6}{RGB}{255,127,0}'))
    doc.packages.append(NoEscape(r'\definecolor{c7}{RGB}{106,61,154}'))
    doc.packages.append(NoEscape(r'\definecolor{c8}{RGB}{51,160,44}'))
    doc.packages.append(NoEscape(r'\definecolor{c9}{RGB}{251,154,153}'))
    doc.packages.append(NoEscape(r'\definecolor{c10}{RGB}{177,89,40}'))
    doc.packages.append(NoEscape(r'\definecolor{c11}{RGB}{202,178,214}'))

    with doc.create(TikZ()) as pic:
        pic.append(NoEscape(r'\pgfplotsset{every tick label/.append style={font=\footnotesize}}'))
        pic.append(NoEscape(r'\begin{groupplot}[group style={group size=4 by 3, horizontal sep=35pt, vertical sep=50pt},height=5cm,width=6.4cm,xmode=log,ymin=0,ymax=1,legend style={font=\tiny,fill=none,draw=none,row sep=-3pt},legend pos=south west,legend cell align=left,label style={font=\footnotesize},y label style={at={(0.08,0.5)}},x label style={at={(0.5,0.05)}}]'))

        nextgrouplot(models, 'res_f_valid',  'Validity',   evaluation_dir)
        nextgrouplot(models, 'res_f_unique', 'Uniqueness', evaluation_dir)
        nextgrouplot(models, 'res_f_novel',  'Novelty',    evaluation_dir)
        nextgrouplot(models, 'res_f_score',  'Score',      evaluation_dir)

        nextgrouplot(models, 'res_t_valid',  'Validity',   evaluation_dir)
        nextgrouplot(models, 'res_t_unique', 'Uniqueness', evaluation_dir)
        nextgrouplot(models, 'res_t_novel',  'Novelty',    evaluation_dir)
        nextgrouplot(models, 'res_t_score',  'Score',      evaluation_dir)

        nextgrouplot(models, 'cor_t_valid',  'Validity',   evaluation_dir)
        nextgrouplot(models, 'cor_t_unique', 'Uniqueness', evaluation_dir)
        nextgrouplot(models, 'cor_t_novel',  'Novelty',    evaluation_dir)
        nextgrouplot(models, 'cor_t_score',  'Score',      evaluation_dir)

        pic.append(NoEscape(r'\end{groupplot}'))

        pic.append(NoEscape(r'\node (t1) at ($(group c2r1.center)!0.5!(group c3r1.center)+(0,2.1cm)$) {Without Resampling (no domain knowledge)};'))
        pic.append(NoEscape(r'\node (t2) at ($(group c2r2.center)!0.5!(group c3r2.center)+(0,2.1cm)$) {With Resampling (no domain knowledge)};'))
        pic.append(NoEscape(r'\node (t3) at ($(group c2r3.center)!0.5!(group c3r3.center)+(0,2.1cm)$) {With Correction (some domain knowledge)};'))

    doc.generate_pdf('train', clean_tex=False)
