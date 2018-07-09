import keras as k
import mxnet as mx
from scipy import misc
import numpy as np


# Test if model predicts the right class for a sample numpy array
def print_indicator(data, model, class_names, bar_width=25):
    X = np.array([data])
    X = k.utils.to_channels_first(X)
    data_iter = mx.io.NDArrayIter(X, None, 1)
    probabilities = model.predict(data_iter)[0]
    prob_array = probabilities.asnumpy()
    left_count = int(prob_array[1] * bar_width)
    right_count = bar_width - left_count
    left_side = '-' * left_count
    right_side = '-' * right_count
    print(class_names[0], left_side + '###' + right_side, class_names[1])

X = np.load('X.npy')
class_names = ['non-smiling', 'smiling']
img = X[-7]

# Load saved keras-mxnet model
sym, arg_params, aux_params = mx.model.load_checkpoint(prefix='smileCNN_model', epoch=0)
model = mx.mod.Module(symbol=sym, 
                    data_names=['/conv2d_1_input1'], 
                    context=mx.cpu(), 
                    label_names=None)
model.bind(for_training=False, 
         data_shapes=[('/conv2d_1_input1', (1,1,32,32))], 
         label_shapes=model._label_shapes)
model.set_params(arg_params, aux_params, allow_missing=True)

print_indicator(img, model, class_names)

