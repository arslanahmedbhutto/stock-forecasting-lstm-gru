"""Baselines and recurrent models. All predict the next-day log return."""

import numpy as np
from sklearn.linear_model import Ridge

RIDGE_ALPHAS = [0.1, 1, 10, 100, 1e3, 1e4, 1e5, 1e6]


class Naive:
    """Tomorrow's close = today's close (predicted return 0)."""

    def fit(self, ds):
        return self

    def predict(self, split):
        return np.zeros(len(split.y))


class Drift:
    """Predict the average training-period daily return every day (always 'up' for AAPL)."""

    def fit(self, ds):
        self.mu = float(ds.train.y.mean())
        return self

    def predict(self, split):
        return np.full(len(split.y), self.mu)


class LinearAR:
    """Ridge regression on the flattened window (an AR model with extra features).
    Alpha is chosen on the validation split."""

    def fit(self, ds):
        Xtr, Xv = _flat(ds.train.X), _flat(ds.val.X)
        best = None
        for a in RIDGE_ALPHAS:
            m = Ridge(alpha=a).fit(Xtr, ds.train.y)
            mse = np.mean((m.predict(Xv) - ds.val.y) ** 2)
            if best is None or mse < best[0]:
                best = (mse, a, m)
        _, self.alpha, self.model = best
        return self

    def predict(self, split):
        return self.model.predict(_flat(split.X))


def _flat(X):
    return X.reshape(len(X), -1)


def build_rnn(cell: str, input_shape, units: int = 64, dropout: float = 0.2, lr: float = 1e-3):
    import tensorflow as tf
    from tensorflow.keras import layers

    Cell = layers.LSTM if cell == "LSTM" else layers.GRU
    model = tf.keras.Sequential([
        layers.Input(shape=input_shape),
        Cell(units),
        layers.Dropout(dropout),
        layers.Dense(16, activation="relu"),
        layers.Dense(1),
    ], name=cell)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mse")
    return model


class RNNForecaster:
    """LSTM / GRU on the scaled feature windows. The target is standardised with
    training mean/std so the loss isn't dominated by tiny raw return values."""

    def __init__(self, cell: str, seed: int, epochs: int = 60, batch_size: int = 64):
        self.cell, self.seed, self.epochs, self.batch_size = cell, seed, epochs, batch_size

    def fit(self, ds):
        import tensorflow as tf

        tf.keras.utils.set_random_seed(self.seed)
        self.mu, self.sd = float(ds.train.y.mean()), float(ds.train.y.std())
        self.model = build_rnn(self.cell, ds.train.X.shape[1:])
        self.history = self.model.fit(
            ds.train.X, (ds.train.y - self.mu) / self.sd,
            validation_data=(ds.val.X, (ds.val.y - self.mu) / self.sd),
            epochs=self.epochs, batch_size=self.batch_size, verbose=0,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8,
                                                        restore_best_weights=True)],
        ).history
        return self

    def predict(self, split):
        return self.model.predict(split.X, verbose=0, batch_size=512).ravel() * self.sd + self.mu

    @property
    def best_val_loss(self):
        return min(self.history["val_loss"])
