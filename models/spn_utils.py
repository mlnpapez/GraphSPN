import torch

def cat2ohe(x, a, num_node_types, num_edge_types):
    x = torch.nn.functional.one_hot(x.to(torch.int64), num_classes=num_node_types)
    a = torch.nn.functional.one_hot(a.to(torch.int64), num_classes=num_edge_types)

    a = torch.movedim(a, -1, 1)

    return x, a

def ohe2cat(x, a):
    x = torch.argmax(x, dim=2)
    a = torch.argmax(a, dim=1)
    return x, a
