import torch
from torch import nn
from torch.distributions import Normal


class Trickster(nn.Module):

    def __init__(self, input_size, hidden_size, n_samples):
        super().__init__()
        self.input_size = input_size
        self.n_samples = n_samples

        self.linear = nn.Linear(input_size, hidden_size)
        self.predictor = nn.Linear(hidden_size, 1)
        self.confident = nn.Linear(hidden_size, 1)

    def forward(self, inputs, targets=None):

        # mlp
        hidden = torch.relu(self.linear(inputs))

        # predicted mean
        predictions = self.predictor(hidden)

        # proxy to std
        confidences = torch.sigmoid(self.confident(hidden))
        # print('c', confidences.item())

        # interpolated mean (w/ hints)
        if self.training:
            inter_preds = confidences * predictions + (1 - confidences) * targets
        else:
            inter_preds = predictions

        # std is a function of confidence
        # c ~ 0: std ~ inf
        # c ~ 1: std ~ 0
        std = -torch.log(confidences)

        # draw (avg over n_samples)
        mean = reparameterize(inter_preds, std, self.n_samples)

        return mean, std


def reparameterize(mu, std, n):
    """
    Given a `mean` and a `std` vectors, returns a sample with gradients.
    Each batch item is given a different `eps` sample from a normal.
    The parameter `n` specifies the number of samples to take and average
    for each batch item.

    Args:
        mu: mean [B x H]
        std: standard deviation [B x H]
        n: number of samples to average

    Returns: an average sample from the distribution [B x H]

    """
    # random sampler
    batch_size = mu.size(0)
    normal = Normal(torch.zeros(n * batch_size), torch.ones(n * batch_size))

    # get one seed per batch item
    # different for each of the n samples
    eps = normal.sample()
    eps = eps.view(n, batch_size, 1).repeat(1, 1, mu.size(1))

    # repeat the mean and std matrices n times
    mu = mu.unsqueeze(0).repeat(n, 1, 1)
    std = std.unsqueeze(0).repeat(n, 1, 1)

    # draw n samples per batch item
    samples = mu + std * eps

    # average over all samples
    avg_samples = samples.sum(0) / n

    return avg_samples


class Vanilla(nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(input_size, hidden_size),
                                   nn.Tanh(),
                                   nn.Linear(hidden_size, input_size))

    def forward(self, x, y=None):
        mean = self.model(x)
        std = torch.zeros(mean.size())
        return mean, std
