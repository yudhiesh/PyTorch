import torch

# ============================================================ #
#                     Tensor Reshaping
# ============================================================ #

x = torch.arange(9)

x_3x3 = x.view(3, 3)
x_3x3 = x.reshape(3, 3)

y = x_3x3.t()
# print(y.view(9))
print(y.contiguous().view(9))

batch = 64
x = torch.rand((batch, 2, 4))
z = x.view(batch, -1)
print(z.shape)

print(x.shape)
# permute() is used to move the axis of the tensor
z = x.permute(0, 2, 1)
print(z.shape)

# Squeeze is the opposite
print(x.unsqueeze(0).shape)
print(x.unsqueeze(1).shape)
# Default adds to the end of the tensor
print(x.unsqueeze(-1).shape)


# Output of how unsqueeze() works
# torch.Size([1, 64, 2, 4])
# torch.Size([64, 1, 2, 4])
# torch.Size([64, 2, 4, 1])
