import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense

X = np.random.randint(
    0,
    128,
    (1000, 50)
)

y = np.eye(128)[X]

model = Sequential([
    Embedding(128, 64),
    Bidirectional(
        LSTM(
            64,
            return_sequences=True
        )
    ),
    Dense(128, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    X,
    y,
    epochs=5,
    batch_size=32
)