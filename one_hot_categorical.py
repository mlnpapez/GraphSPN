import torch
import torch.nn as nn
from torch import distributions as dist

from spn.algorithms.layerwise.distributions import Leaf
from spn.algorithms.layerwise.utils import SamplingContext


class OneHotCategorical(Leaf):
    def __init__(self, in_features: int, out_channels: int, num_repetitions: int=1, dropout=0.0, num_categories: int=4):
        """Creat a one-hot categorical layer.

        Args:
            out_channels: Number of parallel representations for each input feature.
            in_features: Number of input features.
            num_repetitions: Number of parallel repetitions of this layer.

        """
        super().__init__(in_features, out_channels, num_repetitions, dropout)

        # Create categorical parameters
        self.logits = nn.Parameter(torch.randn(1, in_features, out_channels, num_repetitions, num_categories))

    def _get_base_distribution(self):
        return dist.one_hot_categorical.OneHotCategorical(logits=self.logits)


if __name__ == '__main__':
    d = OneHotCategorical(2, 3, 4)
    x = d.sample(10, SamplingContext(10, repetition_indices=torch.zeros(10, dtype=int)))
    print(x.size())
    print(x.unsqueeze(3).size())
    print(d(x.unsqueeze(3)).size())
