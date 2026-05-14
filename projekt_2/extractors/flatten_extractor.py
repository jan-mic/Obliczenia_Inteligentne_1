import numpy as np

class FlattenExtractor:

    def fit(self, x):
        return self

    def transform(self, x):
        x = np.array(x)

        if x.ndim == 4:
            x = x.squeeze(1)

        x_flat = x.reshape(x.shape[0], -1)

        return x_flat

    def fit_transform(self, x):
        return self.transform(x)