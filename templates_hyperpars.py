import os
import json

from datasets import MOLECULAR_DATASETS

NUM_EPOCHS = 40


def template_naive_cat_a(dataset, nd, nk, atom_list,        nl=2, nr=10, ns=10, ni=5,                         batch_size=1000, lr=0.05, seed=0):
    return {
    "dataset": dataset,
    "model": "graphspn_naive_cat_a",
    "model_hyperpars": {
        "nd_n": nd,
        "nk_n": nk,
        "nk_e": 4,
        "nl_n": nl,
        "nl_e": nl,
        "nr_n": nr,
        "nr_e": nr,
        "ns_n": ns,
        "ns_e": ns,
        "ni_n": ni,
        "ni_e": ni,
        "atom_list": atom_list,
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
def template_naive_cat_b(dataset, nd, nk, atom_list,        nl=2, nr=10, ns=10, ni=5,         num_pieces=[2], batch_size=1000, lr=0.05, seed=0):
    return {
    "dataset": dataset,
    "model": "graphspn_naive_cat_b",
    "model_hyperpars": {
        "nd_n": nd,
        "nk_n": nk,
        "nk_e": 4,
        "nl_n": nl,
        "nr_n": nr,
        "ns_n": ns,
        "ns_e": ns,
        "ni_n": ni,
        "ni_e": ni,
        "num_pieces": num_pieces,
        "atom_list": atom_list,
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
def template_naive_cat_c(dataset, nd, nk, atom_list,        nl=2, nr=10, ns=10, ni=5,                         batch_size=1000, lr=0.05, seed=0):
    return {
    "dataset": dataset,
    "model": "graphspn_naive_cat_c",
    "model_hyperpars": {
        "nd_n": nd,
        "nk_n": nk,
        "nk_e": 4,
        "nl_n": nl,
        "nl_e": nl,
        "nr_n": nr,
        "nr_e": nr,
        "ns_n": ns,
        "ns_e": ns,
        "ni_n": ni,
        "ni_e": ni,
        "atom_list": atom_list,
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
def template_naive_cat_d(dataset, nd, nk, atom_list, nc=10, nl=2, nr=10, ns=10, ni=5,                         batch_size=1000, lr=0.05, seed=0):
    return {
    "dataset": dataset,
    "model": "graphspn_naive_cat_d",
    "model_hyperpars": {
        "nc"  : nc,
        "nd_n": nd,
        "nk_n": nk,
        "nk_e": 4,
        "nl_n": nl,
        "nl_e": nl,
        "nr_n": nr,
        "nr_e": nr,
        "ns_n": ns,
        "ns_e": ns,
        "ni_n": ni,
        "ni_e": ni,
        "atom_list": atom_list,
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
def template_naive_cat_e(dataset, nd, nk, atom_list,        nl=2, nr=10, ns=10, ni=5,                         batch_size=1000, lr=0.05, seed=0):
    return {
    "dataset": dataset,
    "model": "graphspn_naive_cat_e",
    "model_hyperpars": {
        "nd_n": nd,
        "nk_n": nk,
        "nk_e": 4,
        "nl": nl,
        "nr": nr,
        "ns": ns,
        "ni": ni,
        "atom_list": atom_list,
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
def template_naive_cat_f(dataset, nd, nk, atom_list,                     ns=10, ni=5,         num_pieces=[2], batch_size=1000, lr=0.05, seed=0):
    return {
    "dataset": dataset,
    "model": "graphspn_naive_cat_f",
    "model_hyperpars": {
        "nd_n": nd,
        "nk_n": nk,
        "nk_e": 4,
        "ns": ns,
        "ni": ni,
        "num_pieces": num_pieces,
        "atom_list": atom_list,
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
def template_naive_cat_g(dataset, nd, nk, atom_list,        nl=2, nr=10, ns=10, ni=5,                         batch_size=1000, lr=0.05, seed=0):
    return {
    "dataset": dataset,
    "model": "graphspn_naive_cat_g",
    "model_hyperpars": {
        "nd_n": nd,
        "nk_n": nk,
        "nk_e": 4,
        "nl": nl,
        "nr": nr,
        "ns": ns,
        "ni": ni,
        "atom_list": atom_list,
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
def template_naive_cat_h(dataset, nd, nk, atom_list, nc=10, nl=2, nr=10, ns=10, ni=5,                         batch_size=1000, lr=0.05, seed=0):
    return {
    "dataset": dataset,
    "model": "graphspn_naive_cat_h",
    "model_hyperpars": {
        "nc"  : nc,
        "nd_n": nd,
        "nk_n": nk,
        "nk_e": 4,
        "nl_n": nl,
        "nl_e": nl,
        "nr_n": nr,
        "nr_e": nr,
        "ns_n": ns,
        "ns_e": ns,
        "ni_n": ni,
        "ni_e": ni,
        "atom_list": atom_list,
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
def template_naive_deq_a(dataset, nd, nk, atom_list,        nl=2, nr=10, ns=10, ni=5, dc=0.1,                 batch_size=1000, lr=0.05, seed=0):
    return {
    "dataset": dataset,
    "model": "graphspn_naive_deq_a",
    "model_hyperpars": {
        "nd_n": nd,
        "nk_n": nk,
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
        "atom_list": atom_list,
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
def template_naive_deq_b(dataset, nd, nk, atom_list,        nl=2, nr=10, ns=10, ni=5, dc=0.1, num_pieces=[2], batch_size=1000, lr=0.05, seed=0):
    return {
    "dataset": dataset,
    "model": "graphspn_naive_deq_b",
    "model_hyperpars": {
        "nd_n": nd,
        "nk_n": nk,
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
        "atom_list": atom_list,
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
def template_naive_deq_c(dataset, nd, nk, atom_list,        nl=2, nr=10, ns=10, ni=5, dc=0.1,                 batch_size=1000, lr=0.05, seed=0):
    return {
    "dataset": dataset,
    "model": "graphspn_naive_deq_c",
    "model_hyperpars": {
        "nd_n": nd,
        "nk_n": nk,
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
        "atom_list": atom_list,
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
def template_naive_deq_d(dataset, nd, nk, atom_list, nc=10, nl=2, nr=10, ns=10, ni=5, dc=0.1,                 batch_size=1000, lr=0.05, seed=0):
    return {
    "dataset": dataset,
    "model": "graphspn_naive_deq_d",
    "model_hyperpars": {
        "nc"  : nc,
        "nd_n": nd,
        "nk_n": nk,
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
        "atom_list": atom_list,
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
def template_naive_deq_h(dataset, nd, nk, atom_list, nc=10, nl=2, nr=10, ns=10, ni=5, dc=0.1,                 batch_size=1000, lr=0.05, seed=0):
    return {
    "dataset": dataset,
    "model": "graphspn_naive_deq_h",
    "model_hyperpars": {
        "nc"  : nc,
        "nd_n": nd,
        "nk_n": nk,
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
        "atom_list": atom_list,
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
    template_naive_deq_h
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
