# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""
`CustomService` defines a MXNet base vision service
"""

from mms.model_service.mxnet_model_service import MXNetBaseService
from mms.utils.mxnet import image, ndarray


class CustomService(MXNetBaseService):
    """CustomService defines a custom service for image classification task.
    In preprocess, input image buffer is read to NDArray and resized respect to input
    shape in signature.
    In post process, top-2 labels are returned.
    """
    def _preprocess(self, data):
        img_list = []
        for idx, img in enumerate(data):
            input_shape = self.signature['inputs'][idx]['data_shape']  # Input shape is NCHW
            [h, w] = input_shape[2:]
            img_arr = image.read(img, 0) #Set flag to 0 for reading grayscale images
            img_arr = image.resize(img_arr, w, h)
            img_arr = image.transform_shape(img_arr)
            img_arr = img_arr.astype('float32') / 256.0 
            img_list.append(img_arr)
        return img_list

    def _postprocess(self, data):
        assert hasattr(self, 'labels'), \
            "Can't find labels attribute. Did you put synset.txt file into " \
            "model archive or manually load class label file in __init__?"
        return [ndarray.top_probability(d, self.labels, top=2) for d in data]
