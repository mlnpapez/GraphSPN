import os
import json

from utils.datasets import MOLECULAR_DATASETS

NUM_EPOCHS = 40


def template_naive_cat_a(dataset, max_atoms, max_types, atom_list,        nl=2, nr=10, ns=10, ni=5,                         batch_size=1000, lr=0.05, seed=0):
    return {
    "dataset": dataset,
    "model": "graphspn_naive_cat_a",
    "model_hyperpars": {
        "nd_n": max_atoms,
        "nk_n": max_types,
        "nk_e": 4,
        "nl_n": nl,
        "nl_e": nl,
        "nr_n": nr,
        "nr_e": nr,
        "ns_n": ns,
        "ns_e": ns,
        "ni_n": ni,
        "ni_e": ni,
        "device": "cpu"
    },
    "optimizer": "adam",
    "optimizer_hyperpars": {
        "lr": lr,
        "betas": [
            0.9,
            0.82
        ]
    },
    "num_epochs": NUM_EPOCHS,
    "batch_size": batch_size,
    "seed": seed
}
def template_naive_cat_b(dataset, max_atoms, max_types, atom_list,        nl=2, nr=10, ns=10, ni=5,         num_pieces=[2], batch_size=1000, lr=0.05, seed=0):
    return {
    "dataset": dataset,
    "model": "graphspn_naive_cat_b",
    "model_hyperpars": {
        "nd_n": max_atoms,
        "nk_n": max_types,
        "nk_e": 4,
        "nl_n": nl,
        "nr_n": nr,
        "ns_n": ns,
        "ns_e": ns,
        "ni_n": ni,
        "ni_e": ni,
        "num_pieces": num_pieces,
        "device": "cpu"
    },
    "optimizer": "adam",
    "optimizer_hyperpars": {
        "lr": lr,
        "betas": [
            0.9,
            0.82
        ]
    },
    "num_epochs": NUM_EPOCHS,
    "batch_size": batch_size,
    "seed": seed
}
def template_naive_cat_c(dataset, max_atoms, max_types, atom_list,        nl=2, nr=10, ns=10, ni=5,                         batch_size=1000, lr=0.05, seed=0):
    return {
    "dataset": dataset,
    "model": "graphspn_naive_cat_c",
    "model_hyperpars": {
        "nd_n": max_atoms,
        "nk_n": max_types,
        "nk_e": 4,
        "nl_n": nl,
        "nl_e": nl,
        "nr_n": nr,
        "nr_e": nr,
        "ns_n": ns,
        "ns_e": ns,
        "ni_n": ni,
        "ni_e": ni,
        "device": "cpu"
    },
    "optimizer": "adam",
    "optimizer_hyperpars": {
        "lr": lr,
        "betas": [
            0.9,
            0.82
        ]
    },
    "num_epochs": NUM_EPOCHS,
    "batch_size": batch_size,
    "seed": seed
}
def template_naive_cat_d(dataset, max_atoms, max_types, atom_list, nc=10, nl=2, nr=10, ns=10, ni=5,                         batch_size=1000, lr=0.05, seed=0):
    return {
    "dataset": dataset,
    "model": "graphspn_naive_cat_d",
    "model_hyperpars": {
        "nc"  : nc,
        "nd_n": max_atoms,
        "nk_n": max_types,
        "nk_e": 4,
        "nl_n": nl,
        "nl_e": nl,
        "nr_n": nr,
        "nr_e": nr,
        "ns_n": ns,
        "ns_e": ns,
        "ni_n": ni,
        "ni_e": ni,
        "device": "cpu"
    },
    "optimizer": "adam",
    "optimizer_hyperpars": {
        "lr": lr,
        "betas": [
            0.9,
            0.82
        ]
    },
    "num_epochs": NUM_EPOCHS,
    "batch_size": batch_size,
    "seed": seed
}
def template_naive_cat_e(dataset, max_atoms, max_types, atom_list,        nl=2, nr=10, ns=10, ni=5,                         batch_size=1000, lr=0.05, seed=0):
    return {
    "dataset": dataset,
    "model": "graphspn_naive_cat_e",
    "model_hyperpars": {
        "nd_n": max_atoms,
        "nk_n": max_types,
        "nk_e": 4,
        "nl": nl,
        "nr": nr,
        "ns": ns,
        "ni": ni,
        "device": "cpu"
    },
    "optimizer": "adam",
    "optimizer_hyperpars": {
        "lr": lr,
        "betas": [
            0.9,
            0.82
        ]
    },
    "num_epochs": NUM_EPOCHS,
    "batch_size": batch_size,
    "seed": seed
}
def template_naive_cat_f(dataset, max_atoms, max_types, atom_list,                     ns=10, ni=5,         num_pieces=[2], batch_size=1000, lr=0.05, seed=0):
    return {
    "dataset": dataset,
    "model": "graphspn_naive_cat_f",
    "model_hyperpars": {
        "nd_n": max_atoms,
        "nk_n": max_types,
        "nk_e": 4,
        "ns": ns,
        "ni": ni,
        "num_pieces": num_pieces,
        "device": "cpu"
    },
    "optimizer": "adam",
    "optimizer_hyperpars": {
        "lr": lr,
        "betas": [
            0.9,
            0.82
        ]
    },
    "num_epochs": NUM_EPOCHS,
    "batch_size": batch_size,
    "seed": seed
}
def template_naive_cat_g(dataset, max_atoms, max_types, atom_list,        nl=2, nr=10, ns=10, ni=5,                         batch_size=1000, lr=0.05, seed=0):
    return {
    "dataset": dataset,
    "model": "graphspn_naive_cat_g",
    "model_hyperpars": {
        "nd_n": max_atoms,
        "nk_n": max_types,
        "nk_e": 4,
        "nl": nl,
        "nr": nr,
        "ns": ns,
        "ni": ni,
        "device": "cpu"
    },
    "optimizer": "adam",
    "optimizer_hyperpars": {
        "lr": lr,
        "betas": [
            0.9,
            0.82
        ]
    },
    "num_epochs": NUM_EPOCHS,
    "batch_size": batch_size,
    "seed": seed
}
def template_naive_cat_h(dataset, max_atoms, max_types, atom_list, nc=10, nl=2, nr=10, ns=10, ni=5,                         batch_size=1000, lr=0.05, seed=0):
    return {
    "dataset": dataset,
    "model": "graphspn_naive_cat_h",
    "model_hyperpars": {
        "nc"  : nc,
        "nd_n": max_atoms,
        "nk_n": max_types,
        "nk_e": 4,
        "nl_n": nl,
        "nl_e": nl,
        "nr_n": nr,
        "nr_e": nr,
        "ns_n": ns,
        "ns_e": ns,
        "ni_n": ni,
        "ni_e": ni,
        "device": "cpu"
    },
    "optimizer": "adam",
    "optimizer_hyperpars": {
        "lr": lr,
        "betas": [
            0.9,
            0.82
        ]
    },
    "num_epochs": NUM_EPOCHS,
    "batch_size": batch_size,
    "seed": seed
}
def template_naive_deq_a(dataset, max_atoms, max_types, atom_list,        nl=2, nr=10, ns=10, ni=5, dc=0.1,                 batch_size=1000, lr=0.05, seed=0):
    return {
    "dataset": dataset,
    "model": "graphspn_naive_deq_a",
    "model_hyperpars": {
        "nd_n": max_atoms,
        "nk_n": max_types,
        "nk_e": 4,
        "nl_n": nl,
        "nl_e": nl,
        "nr_n": nr,
        "nr_e": nr,
        "ns_n": ns,
        "ns_e": ns,
        "ni_n": ni,
        "ni_e": ni,
        "dc_n": dc,
        "dc_e": dc,
        "device": "cpu"
    },
    "optimizer": "adam",
    "optimizer_hyperpars": {
        "lr": lr,
        "betas": [
            0.9,
            0.82
        ]
    },
    "num_epochs": NUM_EPOCHS,
    "batch_size": batch_size,
    "seed": seed
}
def template_naive_deq_b(dataset, max_atoms, max_types, atom_list,        nl=2, nr=10, ns=10, ni=5, dc=0.1, num_pieces=[2], batch_size=1000, lr=0.05, seed=0):
    return {
    "dataset": dataset,
    "model": "graphspn_naive_deq_b",
    "model_hyperpars": {
        "nd_n": max_atoms,
        "nk_n": max_types,
        "nk_e": 4,
        "nl_n": nl,
        "nr_n": nr,
        "ns_n": ns,
        "ns_e": ns,
        "ni_n": ni,
        "ni_e": ni,
        "num_pieces": num_pieces,
        "dc_n": dc,
        "dc_e": dc,
        "device": "cpu"
    },
    "optimizer": "adam",
    "optimizer_hyperpars": {
        "lr": lr,
        "betas": [
            0.9,
            0.82
        ]
    },
    "num_epochs": NUM_EPOCHS,
    "batch_size": batch_size,
    "seed": seed
}
def template_naive_deq_c(dataset, max_atoms, max_types, atom_list,        nl=2, nr=10, ns=10, ni=5, dc=0.1,                 batch_size=1000, lr=0.05, seed=0):
    return {
    "dataset": dataset,
    "model": "graphspn_naive_deq_c",
    "model_hyperpars": {
        "nd_n": max_atoms,
        "nk_n": max_types,
        "nk_e": 4,
        "nl_n": nl,
        "nl_e": nl,
        "nr_n": nr,
        "nr_e": nr,
        "ns_n": ns,
        "ns_e": ns,
        "ni_n": ni,
        "ni_e": ni,
        "dc_n": dc,
        "dc_e": dc,
        "device": "cpu"
    },
    "optimizer": "adam",
    "optimizer_hyperpars": {
        "lr": lr,
        "betas": [
            0.9,
            0.82
        ]
    },
    "num_epochs": NUM_EPOCHS,
    "batch_size": batch_size,
    "seed": seed
}
def template_naive_deq_d(dataset, max_atoms, max_types, atom_list, nc=10, nl=2, nr=10, ns=10, ni=5, dc=0.1,                 batch_size=1000, lr=0.05, seed=0):
    return {
    "dataset": dataset,
    "model": "graphspn_naive_deq_d",
    "model_hyperpars": {
        "nc"  : nc,
        "nd_n": max_atoms,
        "nk_n": max_types,
        "nk_e": 4,
        "nl_n": nl,
        "nl_e": nl,
        "nr_n": nr,
        "nr_e": nr,
        "ns_n": ns,
        "ns_e": ns,
        "ni_n": ni,
        "ni_e": ni,
        "dc_n": dc,
        "dc_e": dc,
        "device": "cpu"
    },
    "optimizer": "adam",
    "optimizer_hyperpars": {
        "lr": lr,
        "betas": [
            0.9,
            0.82
        ]
    },
    "num_epochs": NUM_EPOCHS,
    "batch_size": batch_size,
    "seed": seed
}
def template_naive_deq_h(dataset, max_atoms, max_types, atom_list, nc=10, nl=2, nr=10, ns=10, ni=5, dc=0.1,                 batch_size=1000, lr=0.05, seed=0):
    return {
    "dataset": dataset,
    "model": "graphspn_naive_deq_h",
    "model_hyperpars": {
        "nc"  : nc,
        "nd_n": max_atoms,
        "nk_n": max_types,
        "nk_e": 4,
        "nl_n": nl,
        "nl_e": nl,
        "nr_n": nr,
        "nr_e": nr,
        "ns_n": ns,
        "ns_e": ns,
        "ni_n": ni,
        "ni_e": ni,
        "dc_n": dc,
        "dc_e": dc,
        "device": "cpu"
    },
    "optimizer": "adam",
    "optimizer_hyperpars": {
        "lr": lr,
        "betas": [
            0.9,
            0.82
        ]
    },
    "num_epochs": NUM_EPOCHS,
    "batch_size": batch_size,
    "seed": seed
}


def template_zero_none(dataset, max_atoms, max_types, atom_list, nl=2, nr=10, ns=10, ni=5,          batch_size=1000, lr=0.05, seed=0):
    return {
    "dataset": dataset,
    "model": "graphspn_zero_none",
    "model_hyperpars": {
        "nd_n": max_atoms,
        "nk_n": max_types,
        "nk_e": 4,
        "nl": nl,
        "nr": nr,
        "ns": ns,
        "ni": ni,
        "device": "cuda"
    },
    "optimizer": "adam",
    "optimizer_hyperpars": {
        "lr": lr,
        "betas": [
            0.9,
            0.82
        ]
    },
    "num_epochs": NUM_EPOCHS,
    "batch_size": batch_size,
    "seed": seed
}
def template_zero_full(dataset, max_atoms, max_types, atom_list, nl=2, nr=10, ns=10, ni=5,          batch_size=10,   lr=0.05, seed=0):
    return {
    "dataset": dataset,
    "model": "graphspn_zero_full",
    "model_hyperpars": {
        "nd_n": max_atoms,
        "nk_n": max_types,
        "nk_e": 4,
        "nl": nl,
        "nr": nr,
        "ns": ns,
        "ni": ni,
        "device": "cuda"
    },
    "optimizer": "adam",
    "optimizer_hyperpars": {
        "lr": lr,
        "betas": [
            0.9,
            0.82
        ]
    },
    "num_epochs": NUM_EPOCHS,
    "batch_size": batch_size,
    "seed": seed
}
def template_zero_rand(dataset, max_atoms, max_types, atom_list, nl=2, nr=10, ns=10, ni=5, np=20,   batch_size=1000, lr=0.05, seed=0):
    return {
    "dataset": dataset,
    "model": "graphspn_zero_rand",
    "model_hyperpars": {
        "nd_n": max_atoms,
        "nk_n": max_types,
        "nk_e": 4,
        "nl": nl,
        "nr": nr,
        "ns": ns,
        "ni": ni,
        "np": np,
        "device": "cuda"
    },
    "optimizer": "adam",
    "optimizer_hyperpars": {
        "lr": lr,
        "betas": [
            0.9,
            0.82
        ]
    },
    "num_epochs": NUM_EPOCHS,
    "batch_size": batch_size,
    "seed": seed
}
def template_zero_sort(dataset, max_atoms, max_types, atom_list, nl=2, nr=10, ns=10, ni=5,          batch_size=1000, lr=0.05, seed=0):
    return {
    "dataset": dataset,
    "model": "graphspn_zero_sort",
    "model_hyperpars": {
        "nd_n": max_atoms,
        "nk_n": max_types,
        "nk_e": 4,
        "nl": nl,
        "nr": nr,
        "ns": ns,
        "ni": ni,
        "device": "cuda"
    },
    "optimizer": "adam",
    "optimizer_hyperpars": {
        "lr": lr,
        "betas": [
            0.9,
            0.82
        ]
    },
    "num_epochs": NUM_EPOCHS,
    "batch_size": batch_size,
    "seed": seed
}
def template_zero_kary(dataset, max_atoms, max_types, atom_list, nl=2, nr=10, ns=10, ni=5, arity=5, batch_size=1000, lr=0.05, seed=0):
    return {
    "dataset": dataset,
    "model": "graphspn_zero_kary",
    "model_hyperpars": {
        "nd_n": max_atoms,
        "nk_n": max_types,
        "nk_e": 4,
        "nl": nl,
        "nr": nr,
        "ns": ns,
        "ni": ni,
        "arity": arity,
        "device": "cuda"
    },
    "optimizer": "adam",
    "optimizer_hyperpars": {
        "lr": lr,
        "betas": [
            0.9,
            0.82
        ]
    },
    "num_epochs": NUM_EPOCHS,
    "batch_size": batch_size,
    "seed": seed
}
def template_zero_free(dataset, max_atoms, max_types, atom_list, nl=2, nr=10, ns=10, ni=5,          batch_size=1000, lr=0.05, seed=0):
    return {
    "dataset": dataset,
    "model": "graphspn_zero_free",
    "model_hyperpars": {
        "nd_n": max_atoms,
        "nk_n": max_types,
        "nk_e": 4,
        "nl": nl,
        "nr": nr,
        "ns": ns,
        "ni": ni,
        "device": "cuda"
    },
    "optimizer": "adam",
    "optimizer_hyperpars": {
        "lr": lr,
        "betas": [
            0.9,
            0.82
        ]
    },
    "num_epochs": NUM_EPOCHS,
    "batch_size": batch_size,
    "seed": seed
}


def template_marg_none(dataset, max_atoms, max_types, atom_list, nl=2, nr=10, ns=10, ni=5,          batch_size=1000, lr=0.05, seed=0):
    return {
    "dataset": dataset,
    "model": "graphspn_marg_none",
    "model_hyperpars": {
        "nd_n": max_atoms,
        "nk_n": max_types,
        "nk_e": 4,
        "nl": nl,
        "nr": nr,
        "ns": ns,
        "ni": ni,
        "device": "cuda"
    },
    "optimizer": "adam",
    "optimizer_hyperpars": {
        "lr": lr,
        "betas": [
            0.9,
            0.82
        ]
    },
    "num_epochs": NUM_EPOCHS,
    "batch_size": batch_size,
    "seed": seed
}
def template_marg_full(dataset, max_atoms, max_types, atom_list, nl=2, nr=10, ns=10, ni=5,          batch_size=10,   lr=0.05, seed=0):
    return {
    "dataset": dataset,
    "model": "graphspn_marg_full",
    "model_hyperpars": {
        "nd_n": max_atoms,
        "nk_n": max_types,
        "nk_e": 4,
        "nl": nl,
        "nr": nr,
        "ns": ns,
        "ni": ni,
        "device": "cuda"
    },
    "optimizer": "adam",
    "optimizer_hyperpars": {
        "lr": lr,
        "betas": [
            0.9,
            0.82
        ]
    },
    "num_epochs": NUM_EPOCHS,
    "batch_size": batch_size,
    "seed": seed
}
def template_marg_rand(dataset, max_atoms, max_types, atom_list, nl=2, nr=10, ns=10, ni=5, np=20,   batch_size=1000, lr=0.05, seed=0):
    return {
    "dataset": dataset,
    "model": "graphspn_marg_rand",
    "model_hyperpars": {
        "nd_n": max_atoms,
        "nk_n": max_types,
        "nk_e": 4,
        "nl": nl,
        "nr": nr,
        "ns": ns,
        "ni": ni,
        "np": np,
        "device": "cuda"
    },
    "optimizer": "adam",
    "optimizer_hyperpars": {
        "lr": lr,
        "betas": [
            0.9,
            0.82
        ]
    },
    "num_epochs": NUM_EPOCHS,
    "batch_size": batch_size,
    "seed": seed
}
def template_marg_sort(dataset, max_atoms, max_types, atom_list, nl=2, nr=10, ns=10, ni=5,          batch_size=1000, lr=0.05, seed=0):
    return {
    "dataset": dataset,
    "model": "graphspn_marg_sort",
    "model_hyperpars": {
        "nd_n": max_atoms,
        "nk_n": max_types,
        "nk_e": 4,
        "nl": nl,
        "nr": nr,
        "ns": ns,
        "ni": ni,
        "device": "cuda"
    },
    "optimizer": "adam",
    "optimizer_hyperpars": {
        "lr": lr,
        "betas": [
            0.9,
            0.82
        ]
    },
    "num_epochs": NUM_EPOCHS,
    "batch_size": batch_size,
    "seed": seed
}
def template_marg_kary(dataset, max_atoms, max_types, atom_list, nl=2, nr=10, ns=10, ni=5, arity=5, batch_size=1000, lr=0.05, seed=0):
    return {
    "dataset": dataset,
    "model": "graphspn_marg_kary",
    "model_hyperpars": {
        "nd_n": max_atoms,
        "nk_n": max_types,
        "nk_e": 4,
        "nl": nl,
        "nr": nr,
        "ns": ns,
        "ni": ni,
        "arity": arity,
        "device": "cuda"
    },
    "optimizer": "adam",
    "optimizer_hyperpars": {
        "lr": lr,
        "betas": [
            0.9,
            0.82
        ]
    },
    "num_epochs": NUM_EPOCHS,
    "batch_size": batch_size,
    "seed": seed
}
def template_marg_free(dataset, max_atoms, max_types, atom_list, nl=2, nr=10, ns=10, ni=5,          batch_size=1000, lr=0.05, seed=0):
    return {
    "dataset": dataset,
    "model": "graphspn_marg_free",
    "model_hyperpars": {
        "nd_n": max_atoms,
        "nk_n": max_types,
        "nk_e": 4,
        "nl": nl,
        "nr": nr,
        "ns": ns,
        "ni": ni,
        "device": "cuda"
    },
    "optimizer": "adam",
    "optimizer_hyperpars": {
        "lr": lr,
        "betas": [
            0.9,
            0.82
        ]
    },
    "num_epochs": NUM_EPOCHS,
    "batch_size": batch_size,
    "seed": seed
}


def template_back_none(dataset, max_atoms, max_types, atom_list, nl=2, nz=32, nb=16384, nc=2, batch_size=1000, lr=0.001, seed=0):
    return {
    "dataset": dataset,
    "model": "graphspn_back_none",
    "model_hyperpars": {
        "nd_n": max_atoms,
        "nk_n": max_types,
        "nk_e": 4,
        "nz": nz,
        "nl": nl,
        "nb": nb,
        "nc": nc,
        "device": "cuda"
    },
    "optimizer": "adam",
    "optimizer_hyperpars": {
        "lr": lr,
        "betas": [
            0.8,
            0.7
        ],
        "weight_decay": 1e-4
    },
    "num_epochs": NUM_EPOCHS,
    "batch_size": batch_size,
    "seed": seed
}

def template_moflow(dataset, max_atoms, max_types, atom_list, nf_e=10, nf_n=27, hcha_e=[128, 128], hgnn_n=[64], hlin_n=[128, 64], sq_e=3, batch_size=256, lr=0.001, seed=0):
    return {
    "dataset": dataset,
    "model": "moflow",
    "model_hyperpars": {
        "nk_e": 4,
        "nf_e": nf_e,
        "nb_e": 1,
        "sq_e": sq_e,
        "hcha_e": hcha_e,
        "af_e": "true",
        "lu_e": 1,
        "nd_n": max_atoms,
        "nk_n": max_types,
        "nf_n": nf_n,
        "nb_n": 1,
        "hgnn_n": hgnn_n,
        "hlin_n": hlin_n,
        "mask_row_size_list": [1],
        "mask_row_stride_list": [1],
        "af_n": "true",
        "device": "cuda"
    },
    "optimizer": "adam",
    "optimizer_hyperpars": {
        "lr": lr,
        "weight_decay": 1e-4
    },
    "num_epochs": NUM_EPOCHS,
    "batch_size": batch_size,
    "seed": seed
}


HYPERPARS_TEMPLATES = [
    template_naive_cat_a,
    template_naive_cat_b,
    template_naive_cat_c,
    template_naive_cat_d,
    template_naive_cat_e,
    template_naive_cat_f,
    template_naive_cat_g,
    template_naive_cat_h,

    template_naive_deq_a,
    template_naive_deq_b,
    template_naive_deq_c,
    template_naive_deq_d,
    template_naive_deq_h,

    template_zero_none,
    template_zero_full,
    template_zero_rand,
    template_zero_sort,
    template_zero_kary,

    template_marg_none,
    template_marg_full,
    template_marg_rand,
    template_marg_sort,
    template_marg_kary,

    template_back_none,

    template_moflow,
]


if __name__ == '__main__':
    for dataset, attributes in MOLECULAR_DATASETS.items():
        dir = f'config/{dataset}'
        if os.path.isdir(dir) != True:
            os.makedirs(dir)
        for template in HYPERPARS_TEMPLATES:
            hyperpars = template(**attributes)
            with open(f'{dir}/{hyperpars["model"]}.json', 'w') as f:
                json.dump(hyperpars, f, indent=4)
