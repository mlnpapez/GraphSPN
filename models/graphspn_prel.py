import os
import torch
import torch.nn as nn

from abc import abstractmethod
from einsum import Graph, EinsumNetwork, ExponentialFamilyArray
from torch.distributions import Categorical
from models.spn_utils import cat2ohe, ohe2cat


class GraphSPNNaiveCore(nn.Module):
    def __init__(self,
                 nc,
                 nd_n,
                 nd_e,
                 nk_n,
                 nk_e,
                 ns_n,
                 ns_e,
                 ni_n,
                 ni_e,
                 graph_nodes,
                 graph_edges,
                 device,
                 regime,
                 dc_n=0,
                 dc_e=0
                 ):
        super().__init__()
        self.nd_nodes = nd_n
        self.nd_edges = nd_e
        self.nk_nodes = nk_n
        self.nk_edges = nk_e
        self.regime = regime

        if regime == 'cat':
            ef_dist_n = ExponentialFamilyArray.CategoricalArray
            ef_dist_e = ExponentialFamilyArray.CategoricalArray
            ef_args_n = {'K': nk_n}
            ef_args_e = {'K': nk_e}
            num_dim_n = 1
            num_dim_e = 1
        elif regime == 'deq':
            ef_dist_n = ExponentialFamilyArray.NormalArray
            ef_dist_e = ExponentialFamilyArray.NormalArray
            ef_args_n = {'min_var': 1e-6, 'max_var': 0.01}
            ef_args_e = {'min_var': 1e-6, 'max_var': 0.01}
            num_dim_n = nk_n
            num_dim_e = nk_e
            self.dc_n = dc_n
            self.dc_e = dc_e
        else:
            os.error('Unsupported \'regime\'.')

        args_nodes = EinsumNetwork.Args(
            num_var=nd_n,
            num_dims=num_dim_n,
            num_input_distributions=ni_n,
            num_sums=ns_n,
            num_classes=nc,
            exponential_family=ef_dist_n,
            exponential_family_args=ef_args_n,
            use_em=False)
        args_edges = EinsumNetwork.Args(
            num_var=nd_e,
            num_dims=num_dim_e,
            num_input_distributions=ni_e,
            num_sums=ns_e,
            num_classes=nc,
            exponential_family=ef_dist_e,
            exponential_family_args=ef_args_e,
            use_em=False)

        self.network_nodes = EinsumNetwork.EinsumNetwork(graph_nodes, args_nodes)
        self.network_edges = EinsumNetwork.EinsumNetwork(graph_edges, args_edges)
        self.network_nodes.initialize()
        self.network_edges.initialize()

        self.device = device
        self.to(device)

    @abstractmethod
    def _forward(self, x, a):
        pass

    def forward(self, x, a):
        x = x.to(self.device)
        a = a.to(self.device)
        if   self.regime == 'cat':
            _x, _a = ohe2cat(x, a)
        elif self.regime == 'deq':
            _x = x + self.dc_n*torch.rand(x.size(), device=self.device)
            _a = a + self.dc_e*torch.rand(a.size(), device=self.device)
        else:
            os.error('Unknown regime')
        return self._forward(_x, _a)

    def logpdf(self, x, a):
        return self(x, a).mean()

    @abstractmethod
    def _sample(self, num_samples):
        pass

    def sample(self, num_samples):
        x, a = self._sample(num_samples)
        if self.regime == 'cat':
            x, a = cat2ohe(x, a, self.nk_nodes, self.nk_edges)
        return x, a


class GraphSPNNaiveA(GraphSPNNaiveCore):
    def __init__(self, nd_n, nk_n, nk_e, nl_n, nl_e, nr_n, nr_e, ns_n, ns_e, ni_n, ni_e, regime, dc_n=0.6, dc_e=0.6, device='cuda'):
        nd_e = nd_n**2
        graph_nodes = Graph.random_binary_trees(nd_n, nl_n, nr_n)
        graph_edges = Graph.random_binary_trees(nd_e, nl_e, nr_e)

        super().__init__(1, nd_n, nd_e, nk_n, nk_e, ns_n, ns_e, ni_n, ni_e, graph_nodes, graph_edges, device, regime, dc_n, dc_e)

    def _forward(self, x, a):
        ll_nodes = self.network_nodes(x)
        if   self.regime == 'cat':
            ll_edges = self.network_edges(a.view(-1, self.nd_edges))
        elif self.regime == 'deq':
            a = torch.movedim(a, 1, -1)
            ll_edges = self.network_edges(a.reshape(-1, self.nd_edges, self.nk_edges))
        else:
            os.error('Unknown regime')
        return ll_nodes + ll_edges

    def _sample(self, num_samples):
        x = self.network_nodes.sample(num_samples).cpu()
        a = self.network_edges.sample(num_samples).cpu()
        if   self.regime == 'cat':
            a = a.view(-1, self.nd_nodes, self.nd_nodes)
        elif self.regime == 'deq':
            a = a.view(-1, self.nd_nodes, self.nd_nodes, self.nk_edges)
            a = torch.movedim(a, -1, 1)
        else:
            os.error('Unknown regime')
        return x, a


class GraphSPNNaiveB(GraphSPNNaiveCore):
    def __init__(self, nd_n, nk_n, nk_e, nl_n, nr_n, ns_n, ns_e, ni_n, ni_e, num_pieces, regime, dc_n=0.6, dc_e=0.6, device='cuda'):
        graph_nodes = Graph.random_binary_trees(nd_n, nl_n, nr_n)
        graph_edges = Graph.poon_domingos_structure(shape=[nd_n, nd_n], delta=[[nd_n / d, nd_n / d] for d in num_pieces])

        super().__init__(1, nd_n, nd_n**2, nk_n, nk_e, ns_n, ns_e, ni_n, ni_e, graph_nodes, graph_edges, device, regime, dc_n, dc_e)

    def _forward(self, x, a):
        ll_nodes = self.network_nodes(x)
        if   self.regime == 'cat':
            ll_edges = self.network_edges(a.view(-1, self.nd_edges))
        elif self.regime == 'deq':
            a = torch.movedim(a, 1, -1)
            ll_edges = self.network_edges(a.reshape(-1, self.nd_edges, self.nk_edges))
        else:
            os.error('Unknown regime')
        return ll_nodes + ll_edges

    def _sample(self, num_samples):
        x = self.network_nodes.sample(num_samples).cpu()
        a = self.network_edges.sample(num_samples).cpu()
        if   self.regime == 'cat':
            a = a.view(-1, self.nd_nodes, self.nd_nodes)
        elif self.regime == 'deq':
            a = a.view(-1, self.nd_nodes, self.nd_nodes, self.nk_edges)
            a = torch.movedim(a, -1, 1)
        else:
            os.error('Unknown regime')
        return x, a


class GraphSPNNaiveC(GraphSPNNaiveCore):
    def __init__(self, nd_n, nk_n, nk_e, nl_n, nl_e, nr_n, nr_e, ns_n, ns_e, ni_n, ni_e, regime, dc_n=0.6, dc_e=0.6, device='cuda'):
        nd_e = nd_n * (nd_n - 1) // 2

        graph_nodes = Graph.random_binary_trees(nd_n, nl_n, nr_n)
        graph_edges = Graph.random_binary_trees(nd_e, nl_e, nr_e)

        super().__init__(1, nd_n, nd_e, nk_n, nk_e, ns_n, ns_e, ni_n, ni_e, graph_nodes, graph_edges, device, regime, dc_n, dc_e)

    def _forward(self, x, a):
        ll_nodes = self.network_nodes(x)
        if   self.regime == 'cat':
            m = torch.tril(torch.ones(len(x), self.nd_nodes, self.nd_nodes, dtype=torch.bool), diagonal=-1)
            ll_edges = self.network_edges(a[m].view(-1, self.nd_edges))
        elif self.regime == 'deq':
            m = torch.tril(torch.ones(len(x), self.nk_edges, self.nd_nodes, self.nd_nodes, dtype=torch.bool), diagonal=-1)
            l = a[m].view(-1, self.nk_edges, self.nd_edges)
            ll_edges = self.network_edges(torch.movedim(l, 1, -1))
        else:
            os.error('Unknown regime')
        return ll_nodes + ll_edges

    def _sample(self, num_samples):
        x = self.network_nodes.sample(num_samples).cpu()
        l = self.network_edges.sample(num_samples).cpu()
        if   self.regime == 'cat':
            m = torch.tril(torch.ones(num_samples, self.nd_nodes, self.nd_nodes, dtype=torch.bool), diagonal=-1)
            a = torch.zeros(num_samples, self.nd_nodes, self.nd_nodes)
            a[m] = l.view(num_samples*self.nd_edges)
        elif self.regime == 'deq':
            m = torch.tril(torch.ones(num_samples, self.nk_edges, self.nd_nodes, self.nd_nodes, dtype=torch.bool), diagonal=-1)
            a = torch.zeros(num_samples, self.nk_edges, self.nd_nodes, self.nd_nodes)
            l = torch.movedim(l, -1, 1)
            a[m] = l.reshape(num_samples*self.nd_edges*self.nk_edges)
        else:
            os.error('Unknown regime')

        return x, a


class GraphSPNNaiveD(GraphSPNNaiveCore):
    def __init__(self, nc, nd_n, nk_n, nk_e, nl_n, nl_e, nr_n, nr_e, ns_n, ns_e, ni_n, ni_e, regime, dc_n=0.6, dc_e=0.6, device='cuda'):
        nd_e = nd_n * (nd_n - 1) // 2

        graph_nodes = Graph.random_binary_trees(nd_n, nl_n, nr_n)
        graph_edges = Graph.random_binary_trees(nd_e, nl_e, nr_e)

        super().__init__(nc, nd_n, nd_e, nk_n, nk_e, ns_n, ns_e, ni_n, ni_e, graph_nodes, graph_edges, device, regime, dc_n, dc_e)

        self.weights = nn.Parameter(torch.log_softmax(torch.randn(1, nc, device=self.device), dim=1), requires_grad=True)

    def _forward(self, x, a):
        ll_nodes = self.network_nodes(x)
        if   self.regime == 'cat':
            m = torch.tril(torch.ones(len(x), self.nd_nodes, self.nd_nodes, dtype=torch.bool), diagonal=-1)
            ll_edges = self.network_edges(a[m].view(-1, self.nd_edges))
        elif self.regime == 'deq':
            m = torch.tril(torch.ones(len(x), self.nk_edges, self.nd_nodes, self.nd_nodes, dtype=torch.bool), diagonal=-1)
            l = a[m].view(-1, self.nk_edges, self.nd_edges)
            ll_edges = self.network_edges(torch.movedim(l, 1, -1))
        else:
            os.error('Unknown regime')

        return torch.logsumexp(ll_nodes + ll_edges + torch.log_softmax(self.weights, dim=1), dim=1)

    def _sample(self, num_samples):
        if   self.regime == 'cat':
            x = torch.zeros(num_samples, self.nd_nodes)
            l = torch.zeros(num_samples, self.nd_edges)
        elif self.regime == 'deq':
            x = torch.zeros(num_samples, self.nd_nodes, self.nk_nodes)
            l = torch.zeros(num_samples, self.nd_edges, self.nk_edges)
        else:
            os.error('Unknown regime')

        cs = Categorical(logits=self.weights).sample((num_samples, ))
        for i, c in enumerate(cs):
            x[i, :] = self.network_nodes.sample(1, class_idx=c).cpu()
            l[i, :] = self.network_edges.sample(1, class_idx=c).cpu()

        if   self.regime == 'cat':
            m = torch.tril(torch.ones(num_samples, self.nd_nodes, self.nd_nodes, dtype=torch.bool), diagonal=-1)
            a = torch.zeros(num_samples, self.nd_nodes, self.nd_nodes)
            a[m] = l.view(num_samples*self.nd_edges)
        elif self.regime == 'deq':
            m = torch.tril(torch.ones(num_samples, self.nk_edges, self.nd_nodes, self.nd_nodes, dtype=torch.bool), diagonal=-1)
            a = torch.zeros(num_samples, self.nk_edges, self.nd_nodes, self.nd_nodes)
            l = torch.movedim(l, -1, 1)
            a[m] = l.reshape(num_samples*self.nd_edges*self.nk_edges)
        else:
            os.error('Unknown regime')

        return x, a


class GraphSPNNaiveCatE(nn.Module):
    def __init__(self, nd_n, nk_n, nk_e, ns, ni, nl, nr, device='cuda'):
        super().__init__()
        nd_e = nd_n**2
        self.nd_nodes = nd_n
        self.nd_edges = nd_e
        self.nk_nodes = nk_n
        self.nk_edges = nk_e

        nd = nd_n + nd_e
        nk = max(nk_n, nk_e)

        graph = Graph.random_binary_trees(nd, nl, nr)

        args = EinsumNetwork.Args(
            num_var=nd,
            num_input_distributions=ni,
            num_sums=ns,
            exponential_family=ExponentialFamilyArray.CategoricalArray,
            exponential_family_args={'K': nk},
            use_em=False)

        self.network = EinsumNetwork.EinsumNetwork(graph, args)
        self.network.initialize()

        self.device = device
        self.to(device)

    def forward(self, x, a):
        x, a = ohe2cat(x, a)
        z = torch.cat((x.unsqueeze(1), a), dim=1)
        z = z.view(-1, self.nd_nodes + self.nd_edges).to(self.device)
        return self.network(z)

    def logpdf(self, x, a):
        return self(x, a).mean()

    def sample(self, num_samples):
        z = self.network.sample(num_samples).to(torch.int).cpu()
        z = z.view(-1, self.nd_nodes+1, self.nd_nodes)
        x = z[:, 0 , :]
        a = z[:, 1:, :]
        a[a > 3] = 3
        return cat2ohe(x, a, self.nk_nodes, self.nk_edges)


class GraphSPNNaiveCatF(nn.Module):
    def __init__(self, nd_n, nk_n, nk_e, ns, ni, num_pieces, device='cuda'):
        super().__init__()
        nd_e = nd_n**2
        self.nd_nodes = nd_n
        self.nd_edges = nd_e
        self.nk_nodes = nk_n
        self.nk_edges = nk_e

        nd = nd_n + nd_e
        nk = max(nk_n, nk_e)

        graph = Graph.poon_domingos_structure(shape=[nd_n+1, nd_n], delta=[[(nd_n+1) / d, nd_n / d] for d in num_pieces])

        args = EinsumNetwork.Args(
            num_var=nd,
            num_input_distributions=ni,
            num_sums=ns,
            exponential_family=ExponentialFamilyArray.CategoricalArray,
            exponential_family_args={'K': nk},
            use_em=False)

        self.network = EinsumNetwork.EinsumNetwork(graph, args)
        self.network.initialize()

        self.device = device
        self.to(device)

    def forward(self, x, a):
        x, a = ohe2cat(x, a)
        z = torch.cat((x.unsqueeze(1), a), dim=1)
        z = z.view(-1, self.nd_nodes + self.nd_edges).to(self.device)
        return self.network(z)

    def logpdf(self, x, a):
        return self(x, a).mean()

    def sample(self, num_samples):
        z = self.network.sample(num_samples).to(torch.int).cpu()
        z = z.view(-1, self.nd_nodes+1, self.nd_nodes)
        x = z[:, 0 , :]
        a = z[:, 1:, :]
        a[a > 3] = 3
        return cat2ohe(x, a, self.nk_nodes, self.nk_edges)


class GraphSPNNaiveCatG(nn.Module):
    def __init__(self, nd_n, nk_n, nk_e, ns, ni, nl, nr, device='cuda'):
        super().__init__()
        nd_e = nd_n * (nd_n - 1) // 2

        self.nd_nodes = nd_n
        self.nd_edges = nd_e
        self.nk_nodes = nk_n
        self.nk_edges = nk_e

        nd = nd_n + nd_e
        nk = max(nk_n, nk_e)

        graph = Graph.random_binary_trees(nd, nl, nr)

        args = EinsumNetwork.Args(
            num_var=nd,
            num_input_distributions=ni,
            num_sums=ns,
            exponential_family=ExponentialFamilyArray.CategoricalArray,
            exponential_family_args={'K': nk},
            use_em=False)

        self.network = EinsumNetwork.EinsumNetwork(graph, args)
        self.network.initialize()

        self.device = device
        self.to(device)

    def forward(self, x, a):
        x, a = ohe2cat(x, a)
        a = a.to(self.device)
        m = torch.tril(torch.ones_like(a, dtype=torch.bool), diagonal=-1)
        l = a[m].view(-1, self.nd_edges)
        z = torch.cat((x.to(self.device), l), dim=1)
        return self.network(z)

    def logpdf(self, x, a):
        return self(x, a).mean()

    def sample(self, num_samples):
        z = self.network.sample(num_samples).to(torch.int).cpu()
        x = z[:, :self.nd_nodes]
        l = z[:, self.nd_nodes:]
        m = torch.tril(torch.ones(num_samples, self.nd_nodes, self.nd_nodes, dtype=torch.bool), diagonal=-1)
        a = torch.zeros(num_samples, self.nd_nodes, self.nd_nodes, dtype=torch.int)
        a[m] = l.reshape(num_samples*self.nd_edges)
        a[a > 3] = 3
        return cat2ohe(x, a, self.nk_nodes, self.nk_edges)


class GraphSPNNaiveMixCore(nn.Module):
    def __init__(self, nc, nd_n, nk_n, nk_e, nl_n, nl_e, nr_n, nr_e, ns_n, ns_e, ni_n, ni_e, device, regime):
        super().__init__()
        nd_e = nd_n * (nd_n - 1) // 2

        self.nc = nc

        self.nd_nodes = nd_n
        self.nd_edges = nd_e
        self.nk_nodes = nk_n
        self.nk_edges = nk_e

        self.network_nodes = nn.ParameterList([])
        self.network_edges = nn.ParameterList([])

        for _ in range(nc):
            graph_nodes = Graph.random_binary_trees(nd_n, nl_n, nr_n)
            graph_edges = Graph.random_binary_trees(nd_e, nl_e, nr_e)

            if regime == 'cat':
                ef_dist_n = ExponentialFamilyArray.CategoricalArray
                ef_dist_e = ExponentialFamilyArray.CategoricalArray
                ef_args_n = {'K': nk_n}
                ef_args_e = {'K': nk_e}
                num_dim_n = 1
                num_dim_e = 1
            elif regime == 'deq':
                ef_dist_n = ExponentialFamilyArray.NormalArray
                ef_dist_e = ExponentialFamilyArray.NormalArray
                ef_args_n = {'min_var': 1e-6, 'max_var': 0.01}
                ef_args_e = {'min_var': 1e-6, 'max_var': 0.01}
                num_dim_n = nk_n
                num_dim_e = nk_e
            else:
                os.error('Unsupported \'regime\'.')

            args_nodes = EinsumNetwork.Args(
                num_var=nd_n,
                num_dims=num_dim_n,
                num_input_distributions=ni_n,
                num_sums=ns_n,
                exponential_family=ef_dist_n,
                exponential_family_args=ef_args_n,
                use_em=False)
            args_edges = EinsumNetwork.Args(
                num_var=nd_e,
                num_dims=num_dim_e,
                num_input_distributions=ni_e,
                num_sums=ns_e,
                exponential_family=ef_dist_e,
                exponential_family_args=ef_args_e,
                use_em=False)

            net_n = EinsumNetwork.EinsumNetwork(graph_nodes, args_nodes)
            net_e = EinsumNetwork.EinsumNetwork(graph_edges, args_edges)
            net_n.initialize()
            net_e.initialize()
            net_n.to(device)
            net_e.to(device)

            self.network_nodes.append(net_n)
            self.network_edges.append(net_e)

        self.weights = nn.Parameter(torch.log_softmax(torch.randn(1, nc, device=device), dim=1), requires_grad=True)

        self.device = device
        self.to(device)

    @abstractmethod
    def forward(self, x):
        pass

    def logpdf(self, x, a):
        return self(x, a).mean()

    @abstractmethod
    def sample(self, num_samples):
        pass


class GraphSPNNaiveCatH(GraphSPNNaiveMixCore):
    def __init__(self, nc, nd_n, nk_n, nk_e, nl_n, nl_e, nr_n, nr_e, ns_n, ns_e, ni_n, ni_e, device='cuda'):
        super().__init__(nc, nd_n, nk_n, nk_e, nl_n, nl_e, nr_n, nr_e, ns_n, ns_e, ni_n, ni_e, device, 'cat')

    def forward(self, x, a):
        x, a = ohe2cat(x, a)
        z = x.to(self.device)
        a = a.to(self.device)
        m = torch.tril(torch.ones_like(a, dtype=torch.bool), diagonal=-1)
        l = a[m].view(-1, self.nd_edges)

        ll_nodes = torch.cat([net(z) for net in self.network_nodes], dim=1)
        ll_edges = torch.cat([net(l) for net in self.network_edges], dim=1)

        return torch.logsumexp(ll_nodes + ll_edges + torch.log_softmax(self.weights, dim=1), dim=1)

    def sample(self, num_samples):
        x = torch.zeros(num_samples, self.nd_nodes)
        l = torch.zeros(num_samples, self.nd_edges)

        cs = Categorical(logits=self.weights).sample((num_samples, ))
        for i, c in enumerate(cs):
            x[i, :] = self.network_nodes[c].sample(1).cpu()
            l[i, :] = self.network_edges[c].sample(1).cpu()

        x = x.to(torch.int)
        l = l.to(torch.int)

        m = torch.tril(torch.ones(num_samples, self.nd_nodes, self.nd_nodes, dtype=torch.bool), diagonal=-1)
        a = torch.zeros(num_samples, self.nd_nodes, self.nd_nodes, dtype=torch.int)
        a[m] = l.view(num_samples*self.nd_edges)

        return cat2ohe(x, a, self.nk_nodes, self.nk_edges)


class GraphSPNNaiveDeqH(GraphSPNNaiveMixCore):
    def __init__(self, nc, nd_n, nk_n, nk_e, nl_n, nl_e, nr_n, nr_e, ns_n, ns_e, ni_n, ni_e, dc_n, dc_e, device='cuda'):
        super().__init__(nc, nd_n, nk_n, nk_e, nl_n, nl_e, nr_n, nr_e, ns_n, ns_e, ni_n, ni_e, device, 'deq')

        self.dc_n = dc_n
        self.dc_e = dc_e

    def forward(self, x, a):
        z = x.to(self.device)
        a = a.to(self.device)
        z = z + self.dc_n*torch.rand(z.size(), device=self.device)
        a = a + self.dc_e*torch.rand(a.size(), device=self.device)
        m = torch.tril(torch.ones(self.nd_nodes, self.nd_nodes, dtype=torch.bool), diagonal=-1).unsqueeze(0).unsqueeze(3).expand_as(a)
        l = a[m].view(-1, self.nd_edges, self.nk_edges)

        ll_nodes = torch.cat([net(z) for net in self.network_nodes], dim=1)
        ll_edges = torch.cat([net(l) for net in self.network_edges], dim=1)

        return torch.logsumexp(ll_nodes + ll_edges + torch.log_softmax(self.weights, dim=1), dim=1)

    def sample(self, num_samples):
        x = torch.zeros(num_samples, self.nd_nodes, self.nk_nodes)
        l = torch.zeros(num_samples, self.nd_edges, self.nk_edges)

        cs = Categorical(logits=self.weights).sample((num_samples, ))
        for i, c in enumerate(cs):
            x[i, :] = self.network_nodes[c].sample(1).cpu()
            l[i, :] = self.network_edges[c].sample(1).cpu()

        a = torch.zeros(num_samples, self.nd_nodes, self.nd_nodes, self.nk_edges)
        m = torch.tril(torch.ones(self.nd_nodes, self.nd_nodes, dtype=torch.bool), diagonal=-1).unsqueeze(0).unsqueeze(3).expand_as(a)
        a[m] = l.view(num_samples*self.nd_edges*self.nk_edges)
        a = torch.movedim(a, -1, 1)

        return x, a


MODELS = {
    'graphspn_naive_a': GraphSPNNaiveA,
    'graphspn_naive_b': GraphSPNNaiveB,
    'graphspn_naive_c': GraphSPNNaiveC,
    'graphspn_naive_d': GraphSPNNaiveD,
    'graphspn_naive_cat_e': GraphSPNNaiveCatE,
    'graphspn_naive_cat_f': GraphSPNNaiveCatF,
    'graphspn_naive_cat_g': GraphSPNNaiveCatG,
    'graphspn_naive_cat_h': GraphSPNNaiveCatH,
}
