import itertools

from utils.datasets import MOLECULAR_DATASETS
from utils.templates_hyperpars import *

# LINE = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115, 125, 135, 145]
LINE_SPN = [1, 5, 15, 25, 35, 45, 55, 65]
LINE_MLP = [1, 2, 3, 4, 5, 6, 7]
LINE_FLW = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
NL = 2
NR = 40
NS = 20
NI = 20

def grid_naive_cat_a(dataset):
    nl = [1, 2, 3]
    nr = [1, 10, 40, 80]
    ns = [10, 40, 80]
    ni = [10, 40, 80]
    batch_size = [1000]
    lr = [0.05]
    seed = [0, 1, 2, 3, 4]

    grid = itertools.product(nl, nr, ns, ni, batch_size, lr, seed)

    return [template_naive_cat_a(*dataset.values(), *p) for p in list(grid)]
def grid_naive_cat_b(dataset):
    nl = [1, 2, 3]
    nr = [1, 10, 40, 80]
    ns = [10, 40, 80]
    ni = [10, 40, 80]
    num_pieces = [[2], [4], [2, 2], [4, 4]]
    batch_size = [1000]
    lr = [0.05]
    seed = [0, 1, 2, 3, 4]

    grid = itertools.product(nl, nr, ns, ni, num_pieces, batch_size, lr, seed)

    return [template_naive_cat_b(*dataset.values(), *p) for p in list(grid)]
def grid_naive_cat_c(dataset):
    nl = [1, 2, 3]
    nr = [1, 10, 40, 80]
    ns = [10, 40, 80]
    ni = [10, 40, 80]
    batch_size = [1000]
    lr = [0.05]
    seed = [0, 1, 2, 3, 4]

    grid = itertools.product(nl, nr, ns, ni, batch_size, lr, seed)

    return [template_naive_cat_c(*dataset.values(), *p) for p in list(grid)]
def grid_naive_cat_d(dataset):
    nc = [10, 40, 80]
    nl = [1, 2, 3]
    nr = [1, 10, 40, 80]
    ns = [10, 40, 80]
    ni = [10, 40, 80]
    batch_size = [1000]
    lr = [0.05]
    seed = [0, 1, 2, 3, 4]

    grid = itertools.product(nc, nl, nr, ns, ni, batch_size, lr, seed)

    return [template_naive_cat_d(*dataset.values(), *p) for p in list(grid)]
def grid_naive_cat_e(dataset):
    nl = [1, 2, 3]
    nr = [1, 10, 40, 80]
    ns = [10, 40, 80]
    ni = [10, 40, 80]
    batch_size = [1000]
    lr = [0.05]
    seed = [0, 1, 2, 3, 4]

    grid = itertools.product(nl, nr, ns, ni, batch_size, lr, seed)

    return [template_naive_cat_e(*dataset.values(), *p) for p in list(grid)]
def grid_naive_cat_f(dataset):
    ns = [10, 40, 80]
    ni = [10, 40, 80]
    num_pieces = [[2], [4], [2, 2], [4, 4]]
    batch_size = [1000]
    lr = [0.05]
    seed = [0, 1, 2, 3, 4]

    grid = itertools.product(ns, ni, num_pieces, batch_size, lr, seed)

    return [template_naive_cat_f(*dataset.values(), *p) for p in list(grid)]
def grid_naive_cat_g(dataset):
    nl = [1, 2, 3]
    nr = [1, 10, 40, 80]
    ns = [10, 40, 80]
    ni = [10, 40, 80]
    batch_size = [1000]
    lr = [0.05]
    seed = [0, 1, 2, 3, 4]

    grid = itertools.product(nl, nr, ns, ni, batch_size, lr, seed)

    return [template_naive_cat_g(*dataset.values(), *p) for p in list(grid)]
def grid_naive_cat_h(dataset):
    nc = [10, 40, 80]
    nl = [1, 2, 3]
    nr = [1, 10, 40, 80]
    ns = [10, 40, 80]
    ni = [10, 40, 80]
    batch_size = [1000]
    lr = [0.05]
    seed = [0, 1, 2, 3, 4]

    grid = itertools.product(nc, nl, nr, ns, ni, batch_size, lr, seed)

    return [template_naive_cat_h(*dataset.values(), *p) for p in list(grid)]
def grid_naive_deq_a(dataset):
    nl = [1, 2, 3]
    nr = [1, 10, 40, 80]
    ns = [10, 40, 80]
    ni = [10, 40, 80]
    dc = [0.1, 0.4, 0.9]
    batch_size = [1000]
    lr = [0.05]
    seed = [0, 1, 2, 3, 4]

    grid = itertools.product(nl, nr, ns, ni, dc, batch_size, lr, seed)

    return [template_naive_deq_a(*dataset.values(), *p) for p in list(grid)]
def grid_naive_deq_b(dataset):
    nl = [1, 2, 3]
    nr = [1, 10, 40, 80]
    ns = [10, 40, 80]
    ni = [10, 40, 80]
    num_pieces = [[2], [4], [2, 2], [4, 4]]
    dc = [0.1, 0.4, 0.9]
    batch_size = [1000]
    lr = [0.05]
    seed = [0, 1, 2, 3, 4]

    grid = itertools.product(nl, nr, ns, ni, num_pieces, dc, batch_size, lr, seed)

    return [template_naive_deq_b(*dataset.values(), *p) for p in list(grid)]
def grid_naive_deq_c(dataset):
    nl = [1, 2, 3]
    nr = [1, 10, 40, 80]
    ns = [10, 40, 80]
    ni = [10, 40, 80]
    dc = [0.1, 0.4, 0.9]
    batch_size = [1000]
    lr = [0.05]
    seed = [0, 1, 2, 3, 4]

    grid = itertools.product(nl, nr, ns, ni, dc, batch_size, lr, seed)

    return [template_naive_deq_c(*dataset.values(), *p) for p in list(grid)]
def grid_naive_deq_d(dataset):
    nc = [10, 40, 80]
    nl = [1, 2, 3]
    nr = [1, 10, 40, 80]
    ns = [10, 40, 80]
    ni = [10, 40, 80]
    dc = [0.1, 0.4, 0.9]
    batch_size = [1000]
    lr = [0.05]
    seed = [0, 1, 2, 3, 4]

    grid = itertools.product(nc, nl, nr, ns, ni, dc, batch_size, lr, seed)

    return [template_naive_deq_d(*dataset.values(), *p) for p in list(grid)]
def grid_naive_deq_h(dataset):
    nc = [10, 40, 80]
    nl = [1, 2, 3]
    nr = [1, 10, 40, 80]
    ns = [10, 40, 80]
    ni = [10, 40, 80]
    dc = [0.1, 0.4, 0.9]
    batch_size = [1000]
    lr = [0.05]
    seed = [0, 1, 2, 3, 4]

    grid = itertools.product(nc, nl, nr, ns, ni, dc, batch_size, lr, seed)

    return [template_naive_deq_h(*dataset.values(), *p) for p in list(grid)]


def line_zero_none(dataset):
    return [template_zero_none(*dataset.values(), nl=NL, nr=NR, ns=p, ni=p) for p in LINE_SPN]
def line_zero_full(dataset):
    return [template_zero_full(*dataset.values(), nl=NL, nr=NR, ns=p, ni=p) for p in LINE_SPN]
def line_zero_rand(dataset):
    return [template_zero_rand(*dataset.values(), nl=NL, nr=NR, ns=p, ni=p) for p in LINE_SPN]
def line_zero_sort(dataset):
    return [template_zero_sort(*dataset.values(), nl=NL, nr=NR, ns=p, ni=p) for p in LINE_SPN]
def line_zero_kary(dataset):
    return [template_zero_kary(*dataset.values(), nl=NL, nr=NR, ns=p, ni=p) for p in LINE_SPN]
def line_zero_free(dataset):
    return [template_zero_free(*dataset.values(), nl=NL, nr=NR, ns=p, ni=p) for p in LINE_SPN]


def line_marg_none(dataset):
    return [template_marg_none(*dataset.values(), nl=NL, nr=NR, ns=p, ni=p) for p in LINE_SPN]
def line_marg_full(dataset):
    return [template_marg_full(*dataset.values(), nl=NL, nr=NR, ns=p, ni=p) for p in LINE_SPN]
def line_marg_rand(dataset):
    return [template_marg_rand(*dataset.values(), nl=NL, nr=NR, ns=p, ni=p) for p in LINE_SPN]
def line_marg_sort(dataset):
    return [template_marg_sort(*dataset.values(), nl=NL, nr=NR, ns=p, ni=p) for p in LINE_SPN]
def line_marg_kary(dataset):
    return [template_marg_kary(*dataset.values(), nl=NL, nr=NR, ns=p, ni=p) for p in LINE_SPN]
def line_marg_free(dataset):
    return [template_marg_free(*dataset.values(), nl=NL, nr=NR, ns=p, ni=p) for p in LINE_SPN]


def line_back_none(dataset):
    return [template_back_none(*dataset.values(), nl=p, nz=32, nb=16384, nc=2) for p in LINE_MLP]


def line_moflow(dataset):
    return [template_moflow(*dataset.values(), nf_e=p) for p in LINE_FLW]


GRIDS = {
    'graphspn_naive_cat_a': grid_naive_cat_a,
    'graphspn_naive_cat_b': grid_naive_cat_b,
    'graphspn_naive_cat_c': grid_naive_cat_c,
    'graphspn_naive_cat_d': grid_naive_cat_d,
    'graphspn_naive_cat_e': grid_naive_cat_e,
    'graphspn_naive_cat_f': grid_naive_cat_f,
    'graphspn_naive_cat_g': grid_naive_cat_g,
    'graphspn_naive_cat_h': grid_naive_cat_h,

    'graphspn_naive_deq_a': grid_naive_deq_a,
    'graphspn_naive_deq_b': grid_naive_deq_b,
    'graphspn_naive_deq_c': grid_naive_deq_c,
    'graphspn_naive_deq_d': grid_naive_deq_d,
    'graphspn_naive_deq_h': grid_naive_deq_b,

    'graphspn_zero_none': line_zero_none,
    'graphspn_zero_full': line_zero_full,
    'graphspn_zero_rand': line_zero_rand,
    'graphspn_zero_sort': line_zero_sort,
    'graphspn_zero_kary': line_zero_kary,
    'graphspn_zero_free': line_zero_free,

    'graphspn_marg_none': line_marg_none,
    'graphspn_marg_full': line_marg_full,
    'graphspn_marg_rand': line_marg_rand,
    'graphspn_marg_sort': line_marg_sort,
    'graphspn_marg_kary': line_marg_kary,
    'graphspn_marg_free': line_marg_free,

    'graphspn_back_none': line_back_none,

    'moflow': line_moflow,
}


if __name__ == "__main__":
    for p in grid_naive_cat_a(MOLECULAR_DATASETS['qm9']):
        print(p)
