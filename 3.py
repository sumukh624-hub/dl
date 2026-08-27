import numpy as np


# ============================================================
# INITIAL VALUES
# ============================================================

w = 5.0

lr = 0.1

gradient = 2.0


# ============================================================
# 1. GRADIENT DESCENT
# Formula:
# w = w - learning_rate * gradient
# ============================================================

def gradient_descent(w, gradient, lr):

    w = w - lr * gradient

    return w


# ============================================================
# 2. STOCHASTIC GRADIENT DESCENT
# Formula:
# w = w - learning_rate * gradient
# ============================================================

def sgd(w, gradient, lr):

    w = w - lr * gradient

    return w


# ============================================================
# 3. SGD WITH MOMENTUM
# Formula:
# v = beta * v + learning_rate * gradient
# w = w - v
# ============================================================

def sgd_momentum(
    w,
    gradient,
    lr,
    beta=0.9
):

    v = 0

    v = beta * v + lr * gradient

    w = w - v

    return w


# ============================================================
# 4. ADAGRAD
# Formula:
# G = G + gradient^2
# w = w - lr * gradient / (sqrt(G) + epsilon)
# ============================================================

def adagrad(
    w,
    gradient,
    lr,
    epsilon=1e-8
):

    G = 0

    G = G + gradient ** 2

    w = w - (
        lr * gradient
    ) / (
        np.sqrt(G) + epsilon
    )

    return w


# ============================================================
# 5. RMSPROP
# Formula:
# S = beta*S + (1-beta)*gradient^2
# w = w - lr*gradient / (sqrt(S) + epsilon)
# ============================================================

def rmsprop(
    w,
    gradient,
    lr,
    beta=0.9,
    epsilon=1e-8
):

    S = 0

    S = (
        beta * S
        + (1 - beta) * gradient ** 2
    )

    w = w - (
        lr * gradient
    ) / (
        np.sqrt(S) + epsilon
    )

    return w


# ============================================================
# 6. ADAM
# ============================================================

def adam(
    w,
    gradient,
    lr,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8
):

    # First moment
    m = 0

    # Second moment
    v = 0

    # Time step
    t = 1

    # First moment
    m = (
        beta1 * m
        + (1 - beta1) * gradient
    )

    # Second moment
    v = (
        beta2 * v
        + (1 - beta2) * gradient ** 2
    )

    # Bias correction
    m_hat = m / (
        1 - beta1 ** t
    )

    v_hat = v / (
        1 - beta2 ** t
    )

    # Weight update
    w = w - (
        lr * m_hat
        / (
            np.sqrt(v_hat)
            + epsilon
        )
    )

    return w


# ============================================================
# MAIN PROGRAM
# ============================================================

print("Initial Weight:", w)

print(
    "Gradient Descent:",
    gradient_descent(w, gradient, lr)
)

print(
    "SGD:",
    sgd(w, gradient, lr)
)

print(
    "SGD with Momentum:",
    sgd_momentum(w, gradient, lr)
)

print(
    "AdaGrad:",
    adagrad(w, gradient, lr)
)

print(
    "RMSProp:",
    rmsprop(w, gradient, lr)
)

print(
    "Adam:",
    adam(w, gradient, lr)
)