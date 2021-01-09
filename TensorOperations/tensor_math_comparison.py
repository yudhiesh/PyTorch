import torch

# ============================================================ #
#               Tensor Math & Comparison Operations
# ============================================================ #

x = torch.tensor([2, 4, 6])
y = torch.tensor([1, 3, 8])

# Addition
z1 = torch.empty(3)
print(z1)
print(torch.add(x, y, out=z1))

z2 = torch.add(x, y)
print(z2)
z = x + y
print(z)

# Subtraction
print(torch.sub(y, x))
z_ = y - x
print(z_)

# Division
# This only works if both tensors have the same size
v_ = torch.tensor((4, 3))
# z = torch.true_divide(x, v_)
z = torch.true_divide(x, y)
print(z)

# Inplace operations have the underscore prefix at the end of the method
# It is usually more efficient compared to copying the tensors and then performing the methods on the tensor
t = torch.zeros(3)
print(t)
t.add_(x)
# Another way to do inplace operations
t += x  # t = t + x is not an inplace operations as it copies the tensor
print(t)

# Exponentiation
z = z.pow(x)
print(z)
# Another way to do exponentiation
z = x ** 2
print(z)

# Simple comparison

z = x > 0
print(z)
z = x < 0
print(z)

# Matrix Multiplication
x1 = torch.rand((2, 4))
x2 = torch.rand((4, 2))
x3 = torch.mm(x1, x2)
print(x3)
x2 = x1.mm(x2)
print(x3)

# Matrix Exponentiation
matrix_exp = torch.rand(5, 5)
print(matrix_exp.matrix_power(3))

# element wise multiiplication
z = x * y
print(z)

# dot product
z = torch.dot(z, y)
print(z)

# Batch Matrix Multiplication
batch = 32
n = 10
m = 20
p = 30

tensor1 = torch.rand((batch, n, m))
tensor2 = torch.rand((batch, m, p))

out_bmm = torch.bmm(tensor1, tensor2)  # (batch, n, p)
print(out_bmm)

# Broadcasting
# Even though this is not possible mathematically but PyTorch will handle it for us
x1 = torch.rand((5, 5))
x2 = torch.rand((1, 5))
z = x1 - x2
print(z)

# Other useful tensor operations

sum_x = torch.sum(x, dim=0)
values, indices = torch.max(x, dim=0)
values, indices = torch.min(x, dim=0)
abs_x = torch.abs(x)
z = torch.argmax(x, dim=0)
z = torch.argmin(x, dim=0)
mean_x = torch.mean(x.float(), dim=0)
z = torch.eq(x, y)
print(z)
sorted_y, indices = torch.sort(y, dim=0, descending=False)
# all values below the min are changed to min and the same is for the max
z = torch.clamp(x, min=0, max=4)
print(z)
