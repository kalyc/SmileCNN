import keras as k
from keras.models import model_from_json
model = model_from_json(open('model.json').read())
model.load_weights('weights.h5')
from scipy import misc
import numpy as np
def print_indicator(data, model, class_names, bar_width=25):
    X = np.array([data])
    X = k.utils.to_channels_first(X)
    probabilities = model.predict(X)[0]
    left_count = int(probabilities[1] * bar_width)
    right_count = bar_width - left_count
    left_side = '-' * left_count
    right_side = '-' * right_count
    print(class_names[0], left_side + '###' + right_side, class_names[1])

from utils import show_array
X = np.load('X.npy')
class_names = ['non-smiling', 'smiling']
img = X[-7]
show_array(255 * img)
print_indicator(img, model, class_names)
