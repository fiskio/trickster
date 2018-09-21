import torch
from torch.distributions import Normal


from trickster import Trickster

m = 5

b = 1
input_size = 1
n_samples = 1

input = m * torch.ones(b, 1)
target = m * torch.ones(b, 1, requires_grad=False)

t = Trickster(input_size, n_samples)

optim = torch.optim.Adam(t.parameters(), lr=0.001)


for i in range(100):

    optim.zero_grad()

    mean, std = t(input, target)

    loss = torch.nn.MSELoss()(mean, target) + std
    print(loss.item())

    loss.backward()
    optim.step()
    #print(mean)