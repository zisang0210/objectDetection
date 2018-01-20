# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Tests for yolo_googlenet_feature_extractor."""
import numpy as np
import tensorflow as tf

from object_detection.models import yolo_feature_extractor_test
from object_detection.models import yolo_googlenet_feature_extractor

slim = tf.contrib.slim


class YoloGoogLeNetFeatureExtractorTest(
    yolo_feature_extractor_test.YoloFeatureExtractorTestBase, tf.test.TestCase):

  def _create_feature_extractor(self, depth_multiplier, pad_to_multiple,
                                is_training=True, batch_norm_trainable=True):
    """Constructs a new feature extractor.

    Args:
      depth_multiplier: float depth multiplier for feature extractor
      pad_to_multiple: the nearest multiple to zero pad the input height and
        width dimensions to.
      is_training: whether the network is in training mode.
      batch_norm_trainable: Whether to update batch norm parameters during
        training or not.
    Returns:
      an yolo_meta_arch.YOLOFeatureExtractor object.
    """
    min_depth = 32
    with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm) as sc:
      conv_hyperparams = sc
    return yolo_googlenet_feature_extractor.YOLOGoogLeNetFeatureExtractor(
        is_training, depth_multiplier, min_depth, pad_to_multiple,
        conv_hyperparams, batch_norm_trainable)
  
  def test_end_points_returns_correct_shapes(self):
    image_height = 448
    image_width = 448
    depth_multiplier = 1.0
    pad_to_multiple = 1
    batch_size=4
    expected_endpoints_shapes = {'Conv2d_0a_7x7_s2_64': [batch_size, 224, 224, 64],
                        'MaxPool_0b_2x2': [batch_size, 112, 112, 64],

                        'Conv2d_1a_3x3_192': [batch_size, 112, 112, 192],
                        'MaxPool_1b_2x2': [batch_size, 56, 56, 192],

                        'Conv2d_2a_1x1_128': [batch_size, 56, 56, 128],
                        'Conv2d_2a_3x3_256': [batch_size, 56, 56, 256],
                        'Conv2d_2b_1x1_256': [batch_size, 56, 56, 256],
                        'Conv2d_2b_3x3_512': [batch_size, 56, 56, 512],
                        'MaxPool_2c_2x2': [batch_size, 28,28,512],

                        'Conv2d_3a_a_1x1_256': [batch_size, 28, 28, 256],
                        'Conv2d_3a_a_3x3_512': [batch_size, 28, 28, 512],
                        'Conv2d_3a_b_1x1_256': [batch_size, 28, 28, 256],
                        'Conv2d_3a_b_3x3_512': [batch_size, 28, 28, 512],
                        'Conv2d_3a_c_1x1_256': [batch_size, 28, 28, 256],
                        'Conv2d_3a_c_3x3_512': [batch_size, 28, 28, 512],
                        'Conv2d_3a_d_1x1_256': [batch_size, 28, 28, 256],
                        'Conv2d_3a_d_3x3_512': [batch_size, 28, 28, 512],
                        'Conv2d_3b_1x1_512': [batch_size, 28, 28, 512],
                        'Conv2d_3b_3x3_1024':[batch_size, 28, 28, 1024],
                        'MaxPool_3c_2x2': [batch_size, 14,14,1024],

                        'Conv2d_4a_a_1x1_512': [batch_size, 14, 14, 512],
                        'Conv2d_4a_a_3x3_1024': [batch_size, 14, 14, 1024],
                        'Conv2d_4a_b_1x1_512': [batch_size, 14, 14, 512],
                        'Conv2d_4a_b_3x3_1024': [batch_size, 14, 14, 1024],
                        'Conv2d_4b_3x3_1024': [batch_size, 14, 14, 1024],
                        'Conv2d_4c_3x3_s2_1024':[batch_size, 7, 7, 1024],
                        
                        'Conv2d_5a_3x3_1024': [batch_size, 7, 7, 1024],
                        'Conv2d_5b_3x3_1024': [batch_size, 7, 7, 1024],

                        # 'FullyConnected_6a_4096': 4096,
                        }
    self.check_end_points_returns_correct_shape(
        image_height, image_width, depth_multiplier, pad_to_multiple,
        expected_endpoints_shapes)

  def test_extract_features_returns_correct_shapes_128(self):
    image_height = 448
    image_width = 448
    depth_multiplier = 1.0
    pad_to_multiple = 1
    expected_feature_map_shape = [(4, 7, 7, 1024)]
    self.check_extract_features_returns_correct_shape(
        image_height, image_width, depth_multiplier, pad_to_multiple,
        expected_feature_map_shape)

  # def test_extract_features_returns_correct_shapes_enforcing_min_depth(self):
  #   image_height = 299
  #   image_width = 299
  #   depth_multiplier = 0.5**12
  #   pad_to_multiple = 1
  #   expected_feature_map_shape = [(4, 19, 19, 32), (4, 10, 10, 32),
  #                                 (4, 5, 5, 32), (4, 3, 3, 32),
  #                                 (4, 2, 2, 32), (4, 1, 1, 32)]
  #   self.check_extract_features_returns_correct_shape(
  #       image_height, image_width, depth_multiplier, pad_to_multiple,
  #       expected_feature_map_shape)

  # def test_extract_features_returns_correct_shapes_with_pad_to_multiple(self):
  #   image_height = 299
  #   image_width = 299
  #   depth_multiplier = 1.0
  #   pad_to_multiple = 32
  #   expected_feature_map_shape = [(4, 20, 20, 512), (4, 10, 10, 1024),
  #                                 (4, 5, 5, 512), (4, 3, 3, 256),
  #                                 (4, 2, 2, 256), (4, 1, 1, 128)]
  #   self.check_extract_features_returns_correct_shape(
  #       image_height, image_width, depth_multiplier, pad_to_multiple,
  #       expected_feature_map_shape)

  # def test_extract_features_raises_error_with_invalid_image_size(self):
  #   image_height = 32
  #   image_width = 32
  #   depth_multiplier = 1.0
  #   pad_to_multiple = 1
  #   self.check_extract_features_raises_error_with_invalid_image_size(
  #       image_height, image_width, depth_multiplier, pad_to_multiple)

  # def test_preprocess_returns_correct_value_range(self):
  #   image_height = 128
  #   image_width = 128
  #   depth_multiplier = 1
  #   pad_to_multiple = 1
  #   test_image = np.random.rand(4, image_height, image_width, 3)
  #   feature_extractor = self._create_feature_extractor(depth_multiplier,
  #                                                      pad_to_multiple)
  #   preprocessed_image = feature_extractor.preprocess(test_image)
  #   self.assertTrue(np.all(np.less_equal(np.abs(preprocessed_image), 1.0)))

  def test_variables_only_created_in_scope(self):
    depth_multiplier = 1
    pad_to_multiple = 1
    scope_name = 'GoogLeNet'
    self.check_feature_extractor_variables_under_scope(
        depth_multiplier, pad_to_multiple, scope_name)

  def test_nofused_batchnorm(self):
    image_height = 40
    image_width = 40
    depth_multiplier = 1
    pad_to_multiple = 1
    image_placeholder = tf.placeholder(tf.float32,
                                       [1, image_height, image_width, 3])
    feature_extractor = self._create_feature_extractor(depth_multiplier,
                                                       pad_to_multiple)
    preprocessed_image = feature_extractor.preprocess(image_placeholder)
    _ = feature_extractor.extract_features(preprocessed_image)
    self.assertFalse(any(op.type == 'FusedBatchNorm'
                         for op in tf.get_default_graph().get_operations()))

if __name__ == '__main__':
  tf.test.main()
