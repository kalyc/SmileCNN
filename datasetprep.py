import mxnet as mx
import numpy as np


# Load training images
from utils import list_all_files
negative_paths = list(list_all_files('SMILEsmileD-master/SMILEs/negatives/negatives7/', ['.jpg']))
print('loaded', len(negative_paths), 'negative examples')
positive_paths = list(list_all_files('SMILEsmileD-master/SMILEs/positives/positives7/', ['.jpg']))
print('loaded', len(positive_paths), 'positive examples')
examples = [(path, 0) for path in negative_paths] + [(path, 1) for path in positive_paths]

# Convert loaded images into numpy arrays
def examples_to_dataset(examples, size=32):
    X = []
    y = []
    for path, label in examples:
        img = mx.image.imread(path, flag=0)
        img = mx.image.resize_short(img, size)
        img = img.squeeze().asnumpy()
        X.append(img)
        y.append(label)
    return np.asarray(X), np.asarray(y)

X, y = examples_to_dataset(examples)

# Convert arrays into format consumable by Keras-MXNet
X = X.astype(np.float32) / 255.
y = y.astype(np.int32)
print(X.dtype, X.min(), X.max(), X.shape)
print(y.dtype, y.min(), y.max(), y.shape)

X = np.expand_dims(X, axis=-1)
np.save('X.npy', X)
np.save('y.npy', y)
