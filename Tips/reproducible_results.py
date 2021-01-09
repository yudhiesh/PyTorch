import torch
import numpy as np
import random

seed = 0
# Seed for PyTorch
torch.manual_seed(seed)
# Seed for numpy
np.random.seed(seed)
random.seed(seed)

# if using CUDA
torch.cuda.manual_seed_all(seed)
# the ones below will reduce the performance of the model so only do it when debugging the model
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# With these settings we can make sure that the results are as similar as possible
x = torch.rand((5, 5))
print(torch.einsum("ii->", x))
