import numpy as np

class SymmetryExtractor:

    def fit(self, x):
        return self

    def transform(self, x):

        x = np.array(x)

        if x.ndim == 2:
            x = np.expand_dims(x, axis=0)

        features = []

        for img in x:

            img = img.astype(np.float32)

            left = img[:, :14]
            right = np.fliplr(img[:, 14:])

            vertical_diff = np.abs(left - right)
            vertical_symmetry = 1.0 - (vertical_diff.mean() / 255.0)

            top = img[:14, :]
            bottom = np.flipud(img[14:, :])

            horizontal_diff = np.abs(top - bottom)
            horizontal_symmetry = 1.0 - (horizontal_diff.mean() / 255.0)

            features.append([
                vertical_symmetry,
                horizontal_symmetry
            ])

            print(img.min(), img.max(), img.dtype)
            print(vertical_diff.min(), vertical_diff.max())
            print(vertical_symmetry)

        return np.array(features, dtype=np.float32)

    def fit_transform(self, x):
        return self.transform(x)