import torch

# ============================================================ #
#               Tensor Math & Comparison Operations
# ============================================================ #

batch_size = 25
features = 100

x = torch.rand((batch_size, features))
# To get the features
print(x[0].shape)  # x[0,:]

# To get all the features in the first column
print(x[:, 0].shape)

print(x[2, 0:10].shape)

# Fancy Indexing
x = torch.arange(10)
indices = [2, 5, 6]
print(x[indices])

x = torch.rand((3, 5))
rows = torch.tensor([1, 0])
cols = torch.tensor([4, 0])
print(x[rows, cols])

# More advanced indexing
x = torch.arange(10)
print(x[(x < 2) | (x > 8)])
print(x[x.remainder(2) == 0])

# Useful operations
print(torch.where(x > 5, x, x * 2))
print(x.ndimension())
print(x.numel())
