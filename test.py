import math
import torch
from torch.distributions import Normal, Uniform
from torch.utils.data import TensorDataset, DataLoader

from trickster import Trickster

input_size = 1
hidden_size = 100
n_samples = 10

train_size = 1000
valid_size = 100
epochs = 100000

model = Trickster(input_size, hidden_size, n_samples)

optim = torch.optim.Adam(model.parameters(), lr=0.001)

#
# def two_x(x): return 2 * x
#
#
# def log(x): return math.log(x)


def lobato_dataset(n=20):
    uniform = Uniform(-4, 4)
    normal = Normal(torch.tensor([0.0]), torch.tensor([9.0]))
    x = [uniform.sample().item() for _ in range(n)]
    y = [i ** 3 + normal.sample().item() for i in x]
    return TensorDataset(torch.tensor(x)[:, None], torch.tensor(y)[:, None])


dataset = lobato_dataset()
dataset = DataLoader(dataset, batch_size=1000)
for x, y in dataset:
    print(torch.cat([x, y], dim=1))
    print(x.size())
#
# for x, y in zip(train_x, train_y):
#     print(f'{x:.3f}\t{y:.3f}')

# for e in range(epochs):
#     for x, y in dataset:
#
#         # x = torch.tensor([x]).unsqueeze(0)
#         # y = torch.tensor([y], requires_grad=False).unsqueeze(0)
#
#         optim.zero_grad()
#
#         mean, std = model(x, y)
#         #print(f'mean: {mean.item()}\tstd: {std.item()}')
#
#         loss = torch.nn.MSELoss()(mean, y)# + torch.exp(std)
#         print(f'-> {loss.item()}')
#
#         loss.backward()
#         optim.step()

# with torch.no_grad():
#     all_losses = []
#     for x, y in zip(valid_x, valid_y):
#         x = torch.Tensor([x]).unsqueeze(0)
#         y = torch.Tensor([y]).unsqueeze(0)
#
#         mean, std = model(x, y)
#         loss = torch.nn.MSELoss()(mean, y)
#         all_losses += [loss.item()]
#
#     avg_loss = sum(all_losses) / len(all_losses)
#     print(f'VALID LOSS: {avg_loss}')
