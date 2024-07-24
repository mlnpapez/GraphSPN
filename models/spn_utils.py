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

def permute_graph(xx, aa, pi):
    px = xx[:, pi]
    pa = aa[:, pi, :]
    pa = pa[:, :, pi]
    return px, pa

def flatten_graph(xx, aa, dim=2):
    n = xx.shape[1]
    z = torch.cat((xx.unsqueeze(dim), aa), dim=dim)
    return z.view(-1, n + n**2)

def unflatt_graph(z, nd_nodes, num_full):
    z = z.view(-1, nd_nodes, nd_nodes+1)
    x = z[:, 0:num_full, 0 ]
    a = z[:, 0:num_full, 1:num_full+1]
    return x, a
