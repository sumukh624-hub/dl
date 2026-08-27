import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist


# ============================================================
# LOAD MNIST DATASET
# ============================================================

(X_train, y_train), (X_test, y_test) = mnist.load_data()


# ============================================================
# PREPROCESSING
# ============================================================

X_train = X_train.reshape(-1, 28, 28, 1) / 255.0

X_test = X_test.reshape(-1, 28, 28, 1) / 255.0


# ============================================================
# CREATE CNN MODEL
# ============================================================

model = models.Sequential([

    layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        input_shape=(28, 28, 1)
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    layers.Flatten(),

    layers.Dense(
        64,
        activation="relu"
    ),

    layers.Dense(
        10,
        activation="softmax"
    )
])


# ============================================================
# COMPILE MODEL
# ============================================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# TRAIN MODEL
# ============================================================

model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=64
)


# ============================================================
# TEST MODEL
# ============================================================

print(
    "Accuracy:",
    model.evaluate(X_test, y_test)[1]
)