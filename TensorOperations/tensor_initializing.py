import torch

# ============================================================ #
#                       Initalizing Tensor
# ============================================================ #

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device type : {device.upper()}")
my_tensor = torch.tensor(
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    dtype=torch.float32,
    device=device,
    requires_grad=True,
)
print(my_tensor)
print(my_tensor.dtype)
print(my_tensor.shape)
print(my_tensor.requires_grad)

# Other common intialization methods

# Sort of random intialization
x = torch.empty(size=(3, 3))
print(x)
x = torch.zeros(size=(3, 3))
print(x)
x = torch.rand((3, 3))
print(x)
x = torch.ones((3, 3))
print(x)
x = torch.eye(3, 3)
print(x)
x = torch.arange(0, 10, step=2)
print(x)
x = torch.linspace(start=0.1, end=1, steps=10)
print(x)
# Creating a uniform distribution mean of 0 and std of 1
x = torch.empty((3, 3)).normal_(0, 1)
print(x)
x = torch.empty((3, 3)).uniform_(0, 1)
print(x)
y = torch.rand(3, 3)
print(y)
x = torch.diag(y)
print(x)

# How to initialize and convert tensors to other types
tensor = torch.arange(10)
print(tensor.bool())
# int16
print(tensor.short().dtype)
# int64
print(tensor.long().dtype)
# float16
print(tensor.half().dtype)
# float32
print(tensor.float().dtype)

# Array to tensor and vice-versa
import numpy as np

np_array = np.zeros(10)
print(np_array.dtype)
tensor = torch.from_numpy(np_array)
print(tensor.dtype)
np_array_back = tensor.numpy()
print(np_array_back.dtype)
