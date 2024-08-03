import itertools

from utils.datasets import MOLECULAR_DATASETS
from utils.templates_hyperpars import *

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

def grid_zero_none(dataset):
    nl = [1, 2, 3]
    nr = [10, 40, 80]
    ns = [10, 40, 80]
    ni = [10, 40]
    batch_size = [256]
    lr = [0.05]
    seed = [0, 1, 2, 3, 4]

    grid = itertools.product(nl, nr, ns, ni, batch_size, lr, seed)

    return [template_zero_none(*dataset.values(), *p) for p in list(grid)]
def grid_zero_full(dataset):
    nl = [1, 2, 3]
    nr = [10, 40, 80]
    ns = [10, 40, 80]
    ni = [10, 40]
    batch_size = [256]
    lr = [0.05]
    seed = [0, 1, 2, 3, 4]

    grid = itertools.product(nl, nr, ns, ni, batch_size, lr, seed)

    return [template_zero_full(*dataset.values(), *p) for p in list(grid)]
def grid_zero_rand(dataset):
    nl = [1, 2, 3]
    nr = [10, 40, 80]
    ns = [10, 40, 80]
    ni = [10, 40]
    np = [20]
    batch_size = [256]
    lr = [0.05]
    seed = [0, 1, 2, 3, 4]

    grid = itertools.product(nl, nr, ns, ni, np, batch_size, lr, seed)

    return [template_zero_rand(*dataset.values(), *p) for p in list(grid)]
def grid_zero_sort(dataset):
    nl = [1, 2, 3]
    nr = [10, 40, 80]
    ns = [10, 40, 80]
    ni = [10, 40]
    batch_size = [256]
    lr = [0.05]
    seed = [0, 1, 2, 3, 4]

    grid = itertools.product(nl, nr, ns, ni, batch_size, lr, seed)

    return [template_zero_sort(*dataset.values(), *p) for p in list(grid)]
def grid_zero_kary(dataset):
    nl = [1, 2, 3]
    nr = [10, 40, 80]
    ns = [10, 40, 80]
    ni = [10, 40]
    arity = [2]
    batch_size = [256]
    lr = [0.05]
    seed = [0, 1, 2, 3, 4]

    grid = itertools.product(nl, nr, ns, ni, arity, batch_size, lr, seed)

    return [template_zero_kary(*dataset.values(), *p) for p in list(grid)]
def grid_zero_free(dataset):
    nl = [1, 2, 3]
    nr = [10, 40, 80]
    ns = [10, 40, 80]
    ni = [10, 40]
    batch_size = [256]
    lr = [0.05]
    seed = [0, 1, 2, 3, 4]

    grid = itertools.product(nl, nr, ns, ni, batch_size, lr, seed)

    return [template_zero_free(*dataset.values(), *p) for p in list(grid)]


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


GRIDS = {
    'graphspn_naive_cat_a': grid_naive_cat_a,
    'graphspn_naive_cat_b': grid_naive_cat_b,
    'graphspn_naive_cat_c': grid_naive_cat_c,
    'graphspn_naive_cat_d': grid_naive_cat_d,
    'graphspn_naive_cat_e': grid_naive_cat_e,
    'graphspn_naive_cat_f': grid_naive_cat_f,
    'graphspn_naive_cat_g': grid_naive_cat_g,
    'graphspn_naive_cat_h': grid_naive_cat_h,

    'graphspn_zero_none': grid_zero_none,
    'graphspn_zero_full': grid_zero_full,
    'graphspn_zero_rand': grid_zero_rand,
    'graphspn_zero_sort': grid_zero_sort,
    'graphspn_zero_kary': grid_zero_kary,
    'graphspn_zero_free': grid_zero_free,
}


if __name__ == "__main__":
    for p in grid_naive_cat_a(MOLECULAR_DATASETS['qm9']):
        print(p)
