import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense


# ============================================================
# LOAD WINE DATASETS
# ============================================================

red = pd.read_csv(
    "winequality-red.csv",
    sep=";"
)

white = pd.read_csv(
    "winequality-white.csv",
    sep=";"
)


# ============================================================
# ADD LABELS
# ============================================================

red["type"] = 1

white["type"] = 0


# Combine datasets

wine = pd.concat([
    red,
    white
])


# ============================================================
# INPUT AND OUTPUT
# ============================================================

X = wine.iloc[:, :11]

y = wine["type"]


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)


# ============================================================
# STANDARDIZATION
# ============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(
    X_train
)

X_test = scaler.transform(
    X_test
)


# ============================================================
# NEURAL NETWORK
# ============================================================

model = Sequential([

    Dense(
        12,
        activation="relu",
        input_shape=(11,)
    ),

    Dense(
        8,
        activation="relu"
    ),

    Dense(
        1,
        activation="sigmoid"
    )
])


# ============================================================
# COMPILE
# ============================================================

model.compile(

    optimizer="adam",

    loss="binary_crossentropy",

    metrics=["accuracy"]
)


# ============================================================
# TRAIN
# ============================================================

model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32
)


# ============================================================
# EVALUATE
# ============================================================

loss, acc = model.evaluate(
    X_test,
    y_test
)

print(
    "Accuracy:",
    acc
)