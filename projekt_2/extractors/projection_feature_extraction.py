import numpy as np

class ProjectionFeatureExtractor:
    def __init__(self):
        pass

    def fit(self, x):
        return self

    def transform(self, x):
        return self._extract(x)

    def fit_transform(self, x):
        return self.transform(x)

    def _extract(self, x):
        n = x.shape[0]
        side = int(np.sqrt(x.shape[1]))

        x_img = x.reshape(n, side, side)

        h_proj = np.sum(x_img, axis=2)

        v_proj = np.sum(x_img, axis=1)

        features = np.concatenate([h_proj, v_proj], axis=1)

        return features.astype(np.float32)