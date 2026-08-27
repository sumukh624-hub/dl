import numpy as np


# ============================================================
# 1. Binary Step Activation Function
# ============================================================

def binary_step(x):

    if x >= 0:
        return 1
    else:
        return 0


# ============================================================
# 2. Linear Activation Function
# Formula: f(x) = x
# ============================================================

def linear(x):
    return x


# ============================================================
# 3. Sigmoid Activation Function
# Formula: f(x) = 1 / (1 + e^(-x))
# ============================================================

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ============================================================
# 4. Tanh Activation Function
# Formula:
# f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
# ============================================================

def tanh(x):
    return (
        np.exp(x) - np.exp(-x)
    ) / (
        np.exp(x) + np.exp(-x)
    )


# ============================================================
# 5. ReLU Activation Function
# Formula: f(x) = max(0, x)
# ============================================================

def relu(x):

    if x > 0:
        return x
    else:
        return 0


# ============================================================
# 6. Leaky ReLU Activation Function
# Formula:
# f(x) = x if x > 0
#        0.01x if x <= 0
# ============================================================

def leaky_relu(x):

    if x > 0:
        return x
    else:
        return 0.01 * x


# ============================================================
# 7. ELU Activation Function
# Formula:
# f(x) = x if x > 0
#        alpha(e^x - 1) if x <= 0
# ============================================================

def elu(x, alpha=1):

    if x > 0:
        return x
    else:
        return alpha * (np.exp(x) - 1)


# ============================================================
# 8. Softmax Activation Function
# Formula:
# f(x_i) = e^(x_i) / sum(e^x)
# ============================================================

def softmax(x):

    exp_values = np.exp(
        x - np.max(x)
    )

    return exp_values / np.sum(
        exp_values
    )


# ============================================================
# MAIN PROGRAM
# ============================================================

x = -2

print("Input:", x)
print("Binary Step:", binary_step(x))
print("Linear:", linear(x))
print("Sigmoid:", sigmoid(x))
print("Tanh:", tanh(x))
print("ReLU:", relu(x))
print("Leaky ReLU:", leaky_relu(x))
print("ELU:", elu(x))


# Softmax requires multiple values
values = np.array([1, 2, 3])

print("Softmax:", softmax(values))