"""Re-creation of the original course approach, kept to show why it fails.

Raw closing prices -> MinMaxScaler fitted on the training period -> GRU(64, 32)
predicting the next close. AAPL never traded above $28.55 in the training
period but reaches $327 in the test period, so test inputs are scaled to ~11
instead of [0, 1] and the network's outputs can't follow.
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from stock_forecast.data import LOOKBACK


class OriginalPriceGRU:
    def __init__(self, seed: int, epochs: int = 60):
        self.seed, self.epochs = seed, epochs

    def _windows(self, close_scaled, idx):
        return np.stack([close_scaled[i - LOOKBACK:i] for i in idx])[..., None]

    def fit(self, ds):
        import tensorflow as tf
        from tensorflow.keras import layers

        tf.keras.utils.set_random_seed(self.seed)
        close = ds.frame["Close"].to_numpy("float32").reshape(-1, 1)
        self.scaler = MinMaxScaler().fit(close[: ds.boundaries[0]])
        self.close_scaled = self.scaler.transform(close).ravel()
        self.train_max = float(self.scaler.data_max_[0])

        self.model = tf.keras.Sequential([
            layers.Input(shape=(LOOKBACK, 1)),
            layers.GRU(64, return_sequences=True), layers.Dropout(0.2),
            layers.GRU(32), layers.Dropout(0.2),
            layers.Dense(16, activation="relu"), layers.Dense(1),
        ])
        self.model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
        self.model.fit(
            self._windows(self.close_scaled, ds.train.idx), self.close_scaled[ds.train.idx],
            validation_data=(self._windows(self.close_scaled, ds.val.idx), self.close_scaled[ds.val.idx]),
            epochs=self.epochs, batch_size=32, shuffle=False, verbose=0,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=12,
                                                        restore_best_weights=True)],
        )
        return self

    def predict_price(self, split):
        p = self.model.predict(self._windows(self.close_scaled, split.idx), verbose=0).reshape(-1, 1)
        return self.scaler.inverse_transform(p).ravel()

    def predict(self, split):
        """As a log return, so it can be scored with the same metrics as the other models."""
        return np.log(np.clip(self.predict_price(split), 1e-6, None) / split.prev_close)
