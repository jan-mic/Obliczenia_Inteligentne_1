import numpy as np

class GridFeatureExtractor:

    def __init__(self, grid_size=4):
        self.grid_size = grid_size

    def fit(self, x):
        return self

    def transform(self, x):

        x = np.array(x)

        if x.ndim == 2:
            x = np.expand_dims(x, axis=0)

        features = []

        h, w = x.shape[1], x.shape[2]
        gh = h // self.grid_size
        gw = w // self.grid_size

        for img in x:
            img = img.astype(np.float32)

            vec = []

            for i in range(self.grid_size):
                for j in range(self.grid_size):

                    patch = img[
                        i * gh:(i + 1) * gh,
                        j * gw:(j + 1) * gw
                    ]

                    vec.append(patch.mean() / 255.0)

            features.append(vec)

        return np.array(features, dtype=np.float32)

    def fit_transform(self, x):
        return self.transform(x)