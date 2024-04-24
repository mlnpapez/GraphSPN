import itertools

from datasets import MOLECULAR_DATASETS
from templates_hyperpars import *


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
}


if __name__ == "__main__":
    for p in grid_naive_cat_a(MOLECULAR_DATASETS['qm9']):
        print(p)
