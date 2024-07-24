import math
import torch
import torch.nn as nn
import numpy as np

from scipy import linalg as la
from torch.nn import functional as F


logabs = lambda x: torch.log(torch.abs(x))




class ActNorm(nn.Module):
    def __init__(self, in_channel, logdet=True):
        super().__init__()

        self.loc = nn.Parameter(torch.zeros(1, in_channel, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1, 1))

        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))
        self.logdet = logdet

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input):
        _, _, height, width = input.shape

        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        log_abs = logabs(self.scale)

        logdet = height * width * torch.sum(log_abs)

        if self.logdet:
            return self.scale * (input + self.loc), logdet

        else:
            return self.scale * (input + self.loc)

    def reverse(self, output):
        return output / self.scale - self.loc


class ActNorm2D(nn.Module):
    def __init__(self, in_dim, logdet=True):
        super().__init__()

        self.loc = nn.Parameter(torch.zeros(1, in_dim, 1))
        self.scale = nn.Parameter(torch.ones(1, in_dim, 1))

        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))
        self.logdet = logdet

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .permute(1, 0, 2)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .permute(1, 0, 2)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input):
        _, _, height = input.shape

        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        log_abs = logabs(self.scale)

        logdet = height * torch.sum(log_abs)

        if self.logdet:
            return self.scale * (input + self.loc), logdet

        else:
            return self.scale * (input + self.loc)

    def reverse(self, output):
        return output / self.scale - self.loc


class InvConv2d(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        weight = torch.randn(in_channel, in_channel)
        q, _ = torch.qr(weight)
        weight = q.unsqueeze(2).unsqueeze(3)
        self.weight = nn.Parameter(weight)

    def forward(self, input):
        _, _, height, width = input.shape

        out = F.conv2d(input, self.weight)
        logdet = (
            height * width * torch.slogdet(self.weight.squeeze().double())[1].float()
        )

        return out, logdet

    def reverse(self, output):
        return F.conv2d(
            output, self.weight.squeeze().inverse().unsqueeze(2).unsqueeze(3)
        )


class InvConv2dLU(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        weight = np.random.randn(in_channel, in_channel)
        q, _ = la.qr(weight)
        w_p, w_l, w_u = la.lu(q.astype(np.float32))
        w_s = np.diag(w_u)
        w_u = np.triu(w_u, 1)
        u_mask = np.triu(np.ones_like(w_u), 1)
        l_mask = u_mask.T

        w_p = torch.from_numpy(w_p)
        w_l = torch.from_numpy(w_l)
        w_s = torch.from_numpy(w_s)
        w_u = torch.from_numpy(w_u)

        self.register_buffer('w_p', w_p)
        self.register_buffer('u_mask', torch.from_numpy(u_mask))
        self.register_buffer('l_mask', torch.from_numpy(l_mask))
        self.register_buffer('s_sign', torch.sign(w_s))
        self.register_buffer('l_eye', torch.eye(l_mask.shape[0]))
        self.w_l = nn.Parameter(w_l)
        self.w_s = nn.Parameter(logabs(w_s))
        self.w_u = nn.Parameter(w_u)

    def forward(self, input):
        _, _, height, width = input.shape

        weight = self.calc_weight()

        out = F.conv2d(input, weight)
        logdet = height * width * torch.sum(self.w_s)

        return out, logdet

    def calc_weight(self):
        weight = (
            self.w_p
            @ (self.w_l * self.l_mask + self.l_eye)
            @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s)))
        )

        return weight.unsqueeze(2).unsqueeze(3)

    def reverse(self, output):
        weight = self.calc_weight()

        return F.conv2d(output, weight.squeeze().inverse().unsqueeze(2).unsqueeze(3))





class GraphLinear(nn.Module):
    """Graph Linear layer.
        This function assumes its input is 3-dimensional. Or 4-dim or whatever, only last dim are changed
        Differently from :class:`nn.Linear`, it applies an affine
        transformation to the third axis of input `x`.
        Warning: original Chainer.link.Link use i.i.d. Gaussian initialization as default,
        while default nn.Linear initialization using init.kaiming_uniform_

    .. seealso:: :class:`nn.Linear`
    """
    def __init__(self, in_size, out_size, bias=True):
        super(GraphLinear, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.linear = nn.Linear(in_size, out_size, bias) # Warning: differential initialization from Chainer

    def forward(self, x):
        """Forward propagation.
            Args:
                x (:class:`chainer.Variable`, or :class:`numpy.ndarray`\
                or :class:`cupy.ndarray`):
                    Input array that should be a float array whose ``ndim`` is 3.

                    It represents a minibatch of atoms, each of which consists
                    of a sequence of molecules. Each molecule is represented
                    by integer IDs. The first axis is an index of atoms
                    (i.e. minibatch dimension) and the second one an index
                    of molecules.

            Returns:
                :class:`chainer.Variable`:
                    A 3-dimeisional array.
        """
        h = x
        h = h.reshape(-1, x.shape[-1])  # shape: (s0*s1, s2)
        h = self.linear(h)
        h = h.reshape(tuple(x.shape[:-1] + (self.out_size,)))
        return h


class GraphConv(nn.Module):

    def __init__(self, in_channels, out_channels, num_edge_type=4):
        """

        :param in_channels:   e.g. 8
        :param out_channels:  e.g. 64
        :param num_edge_type:  e.g. 4 types of edges/bonds
        """
        super(GraphConv, self).__init__()

        self.graph_linear_self = GraphLinear(in_channels, out_channels)
        self.graph_linear_edge = GraphLinear(in_channels, out_channels * num_edge_type)
        self.num_edge_type = num_edge_type
        self.in_ch = in_channels
        self.out_ch = out_channels

    def forward(self, adj, h):
        """
        graph convolution over batch and multi-graphs
        :param h: shape: (256,9, 8)
        :param adj: shape: (256,4,9,9)
        :return:
        """
        mb, node, ch = h.shape # 256, 9, 8
        # --- self connection, apply linear function ---
        hs = self.graph_linear_self(h)  # (256,9, 8) --> (256,9, 64)
        # --- relational feature, from neighbor connection ---
        # Expected number of neighbors of a vertex
        # Since you have to divide by it, if its 0, you need to arbitrarily set it to 1
        m = self.graph_linear_edge(h)  # (256,9, 8) --> (256,9, 64*4), namely (256,9, 256)
        m = m.reshape(mb, node, self.out_ch, self.num_edge_type)  # (256,9, 256) --> (256,9, 64, 4)
        m = m.permute(0, 3, 1, 2)  # (256,9, 64, 4) --> (256, 4, 9, 64)
        # m: (batchsize, edge_type, node, ch)
        # hr: (batchsize, edge_type, node, ch)
        hr = torch.matmul(adj, m)  # (256,4,9,9) * (256, 4, 9, 64) = (256, 4, 9, 64)
        # hr: (batchsize, node, ch)
        hr = hr.sum(dim=1)  # (256, 4, 9, 64) --> (256, 9, 64)
        return hs + hr  #




class AffineCoupling(nn.Module):
    def __init__(self, in_channel, hidden_channels, affine=True, mask_swap=False):  # filter_size=512,  --> hidden_channels =(512, 512)
        super(AffineCoupling, self).__init__()

        self.affine = affine
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.mask_swap=mask_swap
        last_h = in_channel // 2
        if affine:
            vh = tuple(hidden_channels) + (in_channel,)
        else:
            vh = tuple(hidden_channels) + (in_channel // 2,)

        for h in vh:
            self.layers.append(nn.Conv2d(last_h, h, kernel_size=3, padding=1))
            self.norms.append(nn.BatchNorm2d(h))
            last_h = h

    def forward(self, input):
        in_a, in_b = input.chunk(2, 1)  # (2,12,32,32) --> (2,6,32,32), (2,6,32,32)

        if self.mask_swap:
            in_a, in_b = in_b, in_a

        if self.affine:
            # (2,12,32,32) --> (2,6,32,32), (2,6,32,32)
            s, t = self._s_t_function(in_a)
            out_b = (in_b + t) * s # (2,6,32,32)
            logdet = torch.sum(torch.log(torch.abs(s)).view(input.shape[0], -1), 1)
        else:
            _, t = self._s_t_function(in_a)
            out_b = in_b + t
            logdet = None

        if self.mask_swap:
            result = torch.cat([out_b, in_a], 1)
        else:
            result = torch.cat([in_a, out_b], 1)

        return result, logdet

    def reverse(self, output):
        out_a, out_b = output.chunk(2, 1)
        if self.mask_swap:
            out_a, out_b = out_b, out_a

        if self.affine:
            s, t = self._s_t_function(out_a)
            in_b = out_b / s - t  # s must not equal to 0
        else:
            _, t = self._s_t_function(out_a)
            in_b = out_b - t

        if self.mask_swap:
            result = torch.cat([in_b, out_a], 1)
        else:
            result = torch.cat([out_a, in_b], 1)

        return result

    def _s_t_function(self, x):
        h = x
        for i in range(len(self.layers)-1):
            h = self.layers[i](h)
            h = self.norms[i](h)
            h = torch.relu(h)
        h = self.layers[-1](h)

        s = None
        if self.affine:
            log_s, t = h.chunk(2, 1)
            s = torch.sigmoid(log_s)
        else:
            t = h
        return s, t


class GraphAffineCoupling(nn.Module):
    def __init__(self, n_node, in_dim, hidden_dim_dict, masked_row, affine=True):
        super(GraphAffineCoupling, self).__init__()
        self.n_node = n_node
        self.in_dim = in_dim
        self.hidden_dim_dict = hidden_dim_dict
        self.masked_row = masked_row
        self.affine = affine

        self.hidden_dim_gnn = hidden_dim_dict['gnn']
        self.hidden_dim_linear = hidden_dim_dict['linear']

        self.net = nn.ModuleList()
        self.norm = nn.ModuleList()
        last_dim = in_dim
        for out_dim in self.hidden_dim_gnn:  # What if use only one gnn???
            self.net.append(GraphConv(last_dim, out_dim))
            self.norm.append(nn.BatchNorm1d(n_node))  # , momentum=0.9 Change norm!!!
            last_dim = out_dim

        self.net_lin = nn.ModuleList()
        self.norm_lin = nn.ModuleList()
        for out_dim in self.hidden_dim_linear:  # What if use only one gnn???
            self.net_lin.append(GraphLinear(last_dim, out_dim))
            self.norm_lin.append(nn.BatchNorm1d(n_node))  # , momentum=0.9 Change norm!!!
            last_dim = out_dim

        if affine:
            self.net_lin.append(GraphLinear(last_dim, in_dim*2))
        else:
            self.net_lin.append(GraphLinear(last_dim, in_dim))

        self.scale = nn.Parameter(torch.zeros(1))  # nn.Parameter(torch.ones(1)) #
        mask = torch.ones(n_node, in_dim)
        mask[masked_row, :] = 0  # masked_row are kept same, and used for _s_t for updating the left rows
        self.register_buffer('mask', mask)

    def forward(self, adj, input):
        masked_x = self.mask * input
        s, t = self._s_t_function(adj, masked_x)  # s must not equal to 0!!!
        if self.affine:
            out = masked_x + (1-self.mask) * (input + t) * s
            logdet = torch.sum(torch.log(torch.abs(s)).view(input.shape[0], -1), 1)  # possibly wrong answer
        else:
            out = masked_x + t*(1-self.mask)
            logdet = None
        return out, logdet

    def reverse(self, adj, output):
        masked_y = self.mask * output
        s, t = self._s_t_function(adj, masked_y)
        if self.affine:
            input = masked_y + (1 - self.mask) * (output / s - t)
        else:
            input = masked_y + (1 - self.mask) * (output - t)
        return input

    def _s_t_function(self, adj, x):
        # adj: (2,4,9,9)  x: # (2,9,5)
        s = None
        h = x
        for i in range(len(self.net)):
            h = self.net[i](adj, h)  # (2,1,9,hidden_dim)
            h = self.norm[i](h)
            h = torch.relu(h)

        for i in range(len(self.net_lin)-1):
            h = self.net_lin[i](h)   # (2,1,9,hidden_dim)
            h = self.norm_lin[i](h)
            h = torch.relu(h)

        h = self.net_lin[-1](h)

        if self.affine:
            log_s, t = h.chunk(2, dim=-1)
            s = torch.sigmoid(log_s)
        else:
            t = h
        return s, t




class Flow(nn.Module):
    def __init__(self, in_channel, hidden_channels, affine=True, conv_lu=2, mask_swap=False):
        super(Flow, self).__init__()

        # More stable to support more flows
        self.actnorm = ActNorm(in_channel)

        if conv_lu == 0:
            self.invconv = InvConv2d(in_channel)
        elif conv_lu == 1:
            self.invconv = InvConv2dLU(in_channel)
        elif conv_lu == 2:
            self.invconv = None
        else:
            raise ValueError("conv_lu in {0,1,2}, 0:InvConv2d, 1:InvConv2dLU, 2:none-just swap to update in coupling")

        self.coupling = AffineCoupling(in_channel, hidden_channels, affine=affine, mask_swap=mask_swap)

    def forward(self, input):
        out, logdet = self.actnorm(input)
        if self.invconv:
            out, det1 = self.invconv(out)
        else:
            det1 = 0
        out, det2 = self.coupling(out)

        logdet = logdet + det1
        if det2 is not None:
            logdet = logdet + det2

        return out, logdet

    def reverse(self, output):
        input = self.coupling.reverse(output)
        if self.invconv:
            input = self.invconv.reverse(input)
        input = self.actnorm.reverse(input)

        return input


class FlowOnGraph(nn.Module):
    def __init__(self, n_node, in_dim, hidden_dim_dict, masked_row, affine=True):
        super(FlowOnGraph, self).__init__()
        self.n_node = n_node
        self.in_dim = in_dim
        self.hidden_dim_dict = hidden_dim_dict
        self.masked_row = masked_row
        self.affine = affine
        self.actnorm = ActNorm2D(in_dim=n_node)
        self.coupling = GraphAffineCoupling(n_node, in_dim, hidden_dim_dict, masked_row, affine=affine)

    def forward(self, adj, input):  # (2,4,9,9) (2,2,9,5)
        out, logdet = self.actnorm(input)
        det1 = 0
        out, det2 = self.coupling(adj, out)

        logdet = logdet + det1
        if det2 is not None:
            logdet = logdet + det2
        return out, logdet

    def reverse(self, adj, output):
        input = self.coupling.reverse(adj, output)
        input = self.actnorm.reverse(input)
        return input


class Block(nn.Module):
    def __init__(self, in_channel, n_flow, squeeze_fold, hidden_channels, affine=True, conv_lu=2):  # in_channel: 3, n_flow: 32
        super(Block, self).__init__()
        # squeeze_fold = 3 for qm9 (bs,4,9,9), squeeze_fold = 2 for zinc (bs, 4, 38, 38)
        #                          (bs,4*3*3,3,3)                        (bs,4*2*2,19,19)
        self.squeeze_fold = squeeze_fold
        squeeze_dim = in_channel * self.squeeze_fold * self.squeeze_fold

        self.flows = nn.ModuleList()
        for i in range(n_flow):
            if conv_lu in (0, 1):
                self.flows.append(Flow(squeeze_dim, hidden_channels,
                                       affine=affine, conv_lu=conv_lu, mask_swap=False))
            else:
                self.flows.append(Flow(squeeze_dim, hidden_channels,
                                       affine=affine, conv_lu=2, mask_swap=bool(i % 2)))

    def forward(self, input):
        out = self._squeeze(input)
        logdet = 0

        for flow in self.flows:
            out, det = flow(out)
            logdet = logdet + det

        out = self._unsqueeze(out)
        return out, logdet

    def reverse(self, output):
        input = self._squeeze(output)

        for flow in self.flows[::-1]:
            input = flow.reverse(input)

        unsqueezed = self._unsqueeze(input)
        return unsqueezed

    def _squeeze(self, x):
        """Trade spatial extent for channels. In forward direction, convert each
        1x4x4 volume of input into a 4x1x1 volume of output.

        Args:
            x (torch.Tensor): Input to squeeze or unsqueeze.
            reverse (bool): Reverse the operation, i.e., unsqueeze.

        Returns:
            x (torch.Tensor): Squeezed or unsqueezed tensor.
        """
        # b, c, h, w = x.size()
        assert len(x.shape) == 4
        b_size, n_channel, height, width = x.shape
        fold = self.squeeze_fold

        squeezed = x.view(b_size, n_channel, height // fold,  fold,  width // fold,  fold)
        squeezed = squeezed.permute(0, 1, 3, 5, 2, 4).contiguous()
        out = squeezed.view(b_size, n_channel * fold * fold, height // fold, width // fold)
        return out

    def _unsqueeze(self, x):
        assert len(x.shape) == 4
        b_size, n_channel, height, width = x.shape
        fold = self.squeeze_fold
        unsqueezed = x.view(b_size, n_channel // (fold * fold), fold, fold, height, width)
        unsqueezed = unsqueezed.permute(0, 1, 4, 2, 5, 3).contiguous()
        out = unsqueezed.view(b_size, n_channel // (fold * fold), height * fold, width * fold)
        return out


class BlockOnGraph(nn.Module):
    def __init__(self, n_node, in_dim, hidden_dim_dict, n_flow, mask_row_size=1, mask_row_stride=1, affine=True):  #, conv_lu=True):
        """

        :param n_node:
        :param in_dim:
        :param hidden_dim:
        :param n_flow:
        :param mask_row_size: number of rows to be masked for update
        :param mask_row_stride: number of steps between two masks' firs row
        :param affine:
        """
        # in_channel=2 deleted. in_channel: 3, n_flow: 32
        super(BlockOnGraph, self).__init__()
        assert 0 < mask_row_size < n_node
        self.flows = nn.ModuleList()
        for i in range(n_flow):
            start = i * mask_row_stride
            masked_row =[r % n_node for r in range(start, start+mask_row_size)]
            self.flows.append(FlowOnGraph(n_node, in_dim, hidden_dim_dict, masked_row=masked_row, affine=affine))

    def forward(self, adj, input):
        out = input
        logdet = 0
        for flow in self.flows:
            out, det = flow(adj, out)
            logdet = logdet + det
        return out, logdet

    def reverse(self, adj, output):
        input = output
        for flow in self.flows[::-1]:
            input = flow.reverse(adj, input)
        return input


class Glow(nn.Module):
    def __init__(self, in_channel, n_flow, n_block, squeeze_fold, hidden_channel, affine=True, conv_lu=2): # in_channel: 3, n_flow:32, n_block:4
        super(Glow, self).__init__()

        self.blocks = nn.ModuleList()
        n_channel = in_channel
        for i in range(n_block):
            self.blocks.append(Block(n_channel, n_flow, squeeze_fold, hidden_channel, affine=affine, conv_lu=conv_lu)) # 3,6,12

    def forward(self, input):
        logdet = 0
        out = input

        for block in self.blocks:
            out, det = block(out)
            logdet = logdet + det

        return out, logdet

    def reverse(self, z):
        h = z
        for i, block in enumerate(self.blocks[::-1]):
            h = block.reverse(h)

        return h


class GlowOnGraph(nn.Module):
    def __init__(self, n_node, in_dim, hidden_dim_dict, n_flow, n_block,
                 mask_row_size_list=[2], mask_row_stride_list=[1], affine=True):  # , conv_lu=True): # in_channel: 2 default
        super(GlowOnGraph, self).__init__()

        assert len(mask_row_size_list) == n_block or len(mask_row_size_list) == 1
        assert len(mask_row_stride_list) == n_block or len(mask_row_stride_list) == 1
        if len(mask_row_size_list) == 1:
            mask_row_size_list = mask_row_size_list * n_block
        if len(mask_row_stride_list) == 1:
            mask_row_stride_list = mask_row_stride_list * n_block
        self.blocks = nn.ModuleList()
        for i in range(n_block):
            mask_row_size = mask_row_size_list[i]
            mask_row_stride = mask_row_stride_list[i]
            self.blocks.append(BlockOnGraph(n_node, in_dim, hidden_dim_dict, n_flow, mask_row_size, mask_row_stride, affine=affine))

    def forward(self, adj, x):
        # adj (bs, 4,9,9), xx:(bs, 9,5)
        logdet = 0
        out = x
        for block in self.blocks:
            out, det = block(adj, out)
            logdet = logdet + det

        return out, logdet

    def reverse(self, adj, z):
        # (bs, 4,9,9), zz: (bs, 9, 5)
        input = z
        for i, block in enumerate(self.blocks[::-1]):
            input = block.reverse(adj, input)

        return input


def rescale_adj(adj, type='all'):
    # Previous paper didn't use rescale_adj.
    # In their implementation, the normalization sum is: num_neighbors = F.sum(adj, axis=(1, 2))
    # In this implementation, the normaliztion term is different
    # raise NotImplementedError
    # (256,4,9, 9):
    # 4: single, double, triple, and bond between disconnected atoms (negative mask of sum of previous)
    # 1-adj[i,:3,:,:].sum(dim=0) == adj[i,4,:,:]
    # usually first 3 matrices have no diagnal, the last has.
    # A_prime = self.A + sp.eye(self.A.shape[0])
    if type == 'view':
        out_degree = adj.sum(dim=-1)
        out_degree_sqrt_inv = out_degree.pow(-1)
        out_degree_sqrt_inv[out_degree_sqrt_inv == float('inf')] = 0
        adj_prime = out_degree_sqrt_inv.unsqueeze(-1) * adj  # (256,4,9,1) * (256, 4, 9, 9) = (256, 4, 9, 9)
    else:  # default type all
        num_neighbors = adj.sum(dim=(1, 2)).float()
        num_neighbors_inv = num_neighbors.pow(-1)
        num_neighbors_inv[num_neighbors_inv == float('inf')] = 0
        adj_prime = num_neighbors_inv[:, None, None, :] * adj
    return adj_prime

def gaussian_nll(x, mean, ln_var, reduce='sum'):
    """Computes the negative log-likelihood of a Gaussian distribution.

    Given two variable ``mean`` representing :math:`\\mu` and ``ln_var``
    representing :math:`\\log(\\sigma^2)`, this function computes in
    elementwise manner the negative log-likelihood of :math:`x` on a
    Gaussian distribution :math:`N(\\mu, S)`,

    .. math::

        -\\log N(x; \\mu, \\sigma^2) =
        \\log\\left(\\sqrt{(2\\pi)^D |S|}\\right) +
        \\frac{1}{2}(x - \\mu)^\\top S^{-1}(x - \\mu),

    where :math:`D` is a dimension of :math:`x` and :math:`S` is a diagonal
    matrix where :math:`S_{ii} = \\sigma_i^2`.

    The output is a variable whose value depends on the value of
    the option ``reduce``. If it is ``'no'``, it holds the elementwise
    loss values. If it is ``'sum'``, loss values are summed up.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.
        mean (:class:`~chainer.Variable` or :ref:`ndarray`): A variable
            representing mean of a Gaussian distribution, :math:`\\mu`.
        ln_var (:class:`~chainer.Variable` or :ref:`ndarray`): A variable
            representing logarithm of variance of a Gaussian distribution,
            :math:`\\log(\\sigma^2)`.
        reduce (str): Reduction option. Its value must be either
            ``'sum'`` or ``'no'``. Otherwise, :class:`ValueError` is raised.

    Returns:
        ~chainer.Variable:
            A variable representing the negative log-likelihood.
            If ``reduce`` is ``'no'``, the output variable holds array
            whose shape is same as one of (hence both of) input variables.
            If it is ``'sum'``, the output variable holds a scalar value.

    """
    if reduce not in ('sum', 'no'):
        raise ValueError(
            "only 'sum' and 'no' are valid for 'reduce', but '%s' is "
            'given' % reduce)

    x_prec = torch.exp(-ln_var)  # 324
    x_diff = x - mean  # (256,324) - (324,) --> (256,324)
    x_power = (x_diff * x_diff) * x_prec * -0.5
    loss = (ln_var + math.log(2 * (math.pi))) / 2 - x_power
    if reduce == 'sum':
        return loss.sum()
    else:
        return loss


class MoFlow(nn.Module):
    def __init__(self,
                 # For bond
                 nk_e=4, nf_e=-1, nb_e=-1, sq_e=-1, hcha_e=None, af_e=True, lu_e=1,
                 # For atom
                 nd_n=-1, nk_n=-1, hgnn_n=None, hlin_n=None, nf_n=-1, nb_n=1, mask_row_size_list=None, mask_row_stride_list=None, af_n=True,
                 # General
                 learn_dist=True, noise_scale=0.6, device='cuda'):
        """
        :param nk_e: Number of bond types/channels
        :param nf_e: Number of masked glow coupling layers per block for bond tensor
        :param nb_e: Number of glow blocks for bond tensor
        :param sq_e:  Squeeze divisor, 3 for qm9, 2 for zinc250k
        :param hcha_e: Hidden channel list for bonds tensor, delimited list input
        :param af_e: Using affine coupling layers for bonds glow
        :param lu_e: Using L-U decomposition trick for 1-1 conv in bonds glow
        :param nd_n: Maximum number of atoms in a molecule
        :param nk_n: Number of atom types
        :param hgnn_n:Hidden dimension list for graph convolution for atoms matrix, delimited list input
        :param hlin_n:Hidden dimension list for linear transformation for atoms, delimited list input
        :param nf_n:Number of masked flow coupling layers per block for atom matrix
        :param nb_n:Number of flow blocks for atom matrix
        :param mask_row_size_list: Mask row list for atom matrix, delimited list input
        :param mask_row_stride_list: Mask row stride  list for atom matrix, delimited list input
        :param af_n: Using affine coupling layers for atom conditional graph flow
        :param path:
        :param learn_dist: learn the distribution of feature matrix
        :param noise_scale:
        """
        super(MoFlow, self).__init__()
        self.b_n_type = nk_e
        self.a_n_node = nd_n
        self.a_n_type = nk_n
        self.b_size = self.a_n_node * self.a_n_node * self.b_n_type
        self.a_size = self.a_n_node * self.a_n_type
        self.noise_scale = noise_scale
        if learn_dist:
            self.ln_var = nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer('ln_var', torch.zeros(1))  # self.ln_var = torch.zeros(1)

        self.bond_model = Glow(
            in_channel=self.b_n_type,  # 4,
            n_flow=nf_e,  # 10, # n_flow 10-->20  n_flow=20
            n_block=nb_e,  # 1,
            squeeze_fold=sq_e,  # 3,
            hidden_channel=hcha_e,  # [128, 128],
            affine=af_e,  # True,
            conv_lu=lu_e  # 0,1,2
        )

        self.atom_model = GlowOnGraph(
            n_node=self.a_n_node,  # 9,
            in_dim=self.a_n_type,  # 5,
            hidden_dim_dict={'gnn': hgnn_n, 'linear': hlin_n},  # {'gnn': [64], 'linear': [128, 64]},
            n_flow=nf_n,  # 27,
            n_block=nb_n,  # 1,
            mask_row_size_list=mask_row_size_list,  # [1],
            mask_row_stride_list=mask_row_stride_list,  # [1],
            affine=af_n  # True
        )

        self.device = device
        self.to(device)

    def forward(self, adj, x):
        """
        :param adj:  (256,4,9,9)
        :param x: (256,9,5)
        :return:
        """
        x = x.to(torch.float32)
        x = x.to(self.device)
        adj = adj.to(torch.float32)
        adj = adj.to(self.device)

        adj_normalized = rescale_adj(adj)
        h = x  # (256,9,5)
        # add uniform noise to node feature matrices
        # + noise didn't change log-det. 1. change to logit transform 2. *0.9 ---> *other value???
        if self.training:
            if self.noise_scale == 0:
                h = h/2.0 - 0.5 + torch.rand_like(x) * 0.4  #/ 2.0  similar to X + U(0, 0.8)   *0.5*0.8=0.4
            else:
                h = h + torch.rand_like(x) * self.noise_scale  # noise_scale default 0.9
        h, sum_log_det_jacs_x = self.atom_model(adj_normalized, h)

        # add uniform noise to adjacency tensors
        if self.training:
            if self.noise_scale == 0:
                adj = adj/2.0 - 0.5 + torch.rand_like(adj) * 0.4  #/ 2.0
            else:
                adj = adj + torch.rand_like(adj) * self.noise_scale  # (256,4,9,9) noise_scale default 0.9
        adj_h, sum_log_det_jacs_adj = self.bond_model(adj)
        out = [h, adj_h]  # combine to one tensor later bs * dim tensor

        return out, [sum_log_det_jacs_x, sum_log_det_jacs_adj]

    def reverse(self, z, true_adj=None):  # change!!! z[0] --> for z_x, z[1] for z_adj, a list!!!
        """
        Returns a molecule, given its latent vector.
        :param z: latent vector. Shape: [B, N*N*M + N*T]    (100,369) 369=9*9 * 4 + 9*5
            B = Batch size, N = number of atoms, M = number of bond types,
            T = number of atom types (Carbon, Oxygen etc.)
        :param true_adj: used for testing. An adjacency matrix of a real molecule
        :return: adjacency matrix and feature matrix of a molecule
        """
        batch_size = z.shape[0]  # 100,  z.shape: (100,369)

        with torch.no_grad():
            z_x = z[:, :self.a_size]  # (100, 45)
            z_adj = z[:, self.a_size:]  # (100, 324)

            if true_adj is None:
                h_adj = z_adj.reshape(batch_size, self.b_n_type, self.a_n_node, self.a_n_node)  # (100,4,9,9)
                h_adj = self.bond_model.reverse(h_adj)

                if self.noise_scale == 0:
                    h_adj = (h_adj + 0.5) * 2
                # decode adjacency matrix from h_adj
                adj = h_adj
                adj = adj + adj.permute(0, 1, 3, 2)
                adj = adj / 2
                adj = adj.softmax(dim=1)  # (100,4!!!,9,9) prob. for edge 0-3 for every pair of nodes
                max_bond = adj.max(dim=1).values.reshape(batch_size, -1, self.a_n_node, self.a_n_node)  # (100,1,9,9)
                adj = torch.floor(adj / max_bond)  # (100,4,9,9) /  (100,1,9,9) -->  (100,4,9,9)
            else:
                adj = true_adj

            h_x = z_x.reshape(batch_size, self.a_n_node, self.a_n_type)
            adj_normalized = rescale_adj(adj).to(h_x)
            h_x = self.atom_model.reverse(adj_normalized, h_x)
            if self.noise_scale == 0:
                h_x = (h_x + 0.5) * 2
            # h_x = torch.sigmoid(h_x)  # to delete for logit
        return adj, h_x  # (100,4,9,9), (100,9,5)

    def log_prob(self, z, logdet):  # z:[(256,45), (256,324)] logdet:[(256,),(256,)]
        # If I din't use self.ln_var, then I can parallel the code!
        z[0] = z[0].reshape(z[0].shape[0],-1)
        z[1] = z[1].reshape(z[1].shape[0], -1)

        logdet[0] = logdet[0] - self.a_size * math.log(2.)  # n_bins = 2**n_bit = 2**1=2
        logdet[1] = logdet[1] - self.b_size * math.log(2.)
        if len(self.ln_var) == 1:
            ln_var_adj = self.ln_var * torch.ones([self.b_size]).to(z[0])  # (324,)
            ln_var_x = self.ln_var * torch.ones([self.a_size]).to(z[0])  # (45)
        else:
            ln_var_adj = self.ln_var[0] * torch.ones([self.b_size]).to(z[0])  # (324,) 0 for bond
            ln_var_x = self.ln_var[1] * torch.ones([self.a_size]).to(z[0])  # (45) 1 for atom
        nll_adj = torch.mean(
            torch.sum(gaussian_nll(z[1], torch.zeros(self.b_size).to(z[0]), ln_var_adj, reduce='no'), dim=1)
            - logdet[1])
        nll_adj = nll_adj / (self.b_size * math.log(2.))  # the negative log likelihood per dim with log base 2

        nll_x = torch.mean(torch.sum(
            gaussian_nll(z[0], torch.zeros(self.a_size).to(z[0]), ln_var_x, reduce='no'),
            dim=1) - logdet[0])
        nll_x = nll_x / (self.a_size * math.log(2.))  # the negative log likelihood per dim with log base 2
        if nll_x.item() < 0:
            print('nll_x:{}'.format(nll_x.item()))

        return [nll_x, nll_adj]

    def logpdf(self, x, a):
        z, sum_log_det_jacs = self(a, x)
        nll = self.log_prob(z, sum_log_det_jacs)
        loss = nll[0] + nll[1]

        return -loss

    def sample(self, num_samples):
        z_dim = self.b_size + self.a_size
        mu = np.zeros(z_dim)
        sigma_diag = np.ones(z_dim)
        sigma = 0.7 * sigma_diag
        z = np.random.normal(mu, sigma, (num_samples, z_dim))  # .astype(np.float32)
        z = torch.from_numpy(z).float().to(self.device)
        a, x = self.reverse(z)

        return x, a
