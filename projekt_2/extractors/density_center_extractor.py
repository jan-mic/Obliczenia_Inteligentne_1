import numpy as np

class DensityCenterExtractor:

    def fit(self, x):
        return self

    def transform(self, x):
        x = np.array(x)

        if x.ndim == 2:
            x = np.expand_dims(x, axis=0)

        features = []

        for image in x:

            active_pixels = image > 0

            density = np.sum(active_pixels)

            ys, xs = np.where(active_pixels)

            if len(ys) == 0:
                center_y = 0.0
            else:
                center_y = np.mean(ys)

            features.append([density, center_y])

        return np.array(features, dtype=np.float32)

    def fit_transform(self, x):
        return self.transform(x)