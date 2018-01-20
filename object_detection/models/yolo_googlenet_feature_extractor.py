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

"""YOLOFeatureExtractor for GoogLeNet features."""

import tensorflow as tf

from object_detection.meta_architectures import yolo_meta_arch
from object_detection.models import feature_map_generators
from object_detection.utils import ops
from collections import namedtuple

slim = tf.contrib.slim

# Conv and Insert1x1Conv namedtuple define layers of the GoogLeNet architecture
# Conv defines 3x3 convolution layers
# Insert1x1Conv defines 3x3 convolution followed by 1x1 convolution.
# stride is the stride of the convolution
# depth is the number of channels or filters in a layer
Conv = namedtuple('Conv', ['kernel', 'stride', 'depth'])
Insert1x1Conv = namedtuple('Insert1x1Conv', ['kernel', 'stride', 'depth'])
Insert1x1ConvRep = namedtuple('Insert1x1ConvRep', ['kernel', 'stride', 'depth','rep'])
Maxpool = namedtuple('Maxpool', ['kernel', 'stride'])
FC = namedtuple('FC', ['output'])

# _CONV_DEFS specifies the GoogLeNet body
_CONV_DEFS = [
    [Conv(kernel=[7, 7], stride=2, depth=64),
    Maxpool(kernel=[2, 2], stride=2)],

    [Conv(kernel=[3, 3], stride=1, depth=192),
    Maxpool(kernel=[2, 2], stride=2)],

    [Insert1x1Conv(kernel=[3, 3], stride=1, depth=256),
    Insert1x1Conv(kernel=[3, 3], stride=1, depth=512),
    Maxpool(kernel=[2, 2], stride=2)],

    [Insert1x1ConvRep(kernel=[3, 3], stride=1, depth=512,rep=4),
    Insert1x1Conv(kernel=[3, 3], stride=1, depth=1024),
    Maxpool(kernel=[2, 2], stride=2)],

    [Insert1x1ConvRep(kernel=[3, 3], stride=1, depth=1024,rep=2),
    Conv(kernel=[3, 3], stride=1, depth=1024),
    Conv(kernel=[3, 3], stride=2, depth=1024)],

    [Conv(kernel=[3, 3], stride=1, depth=1024),
    Conv(kernel=[3, 3], stride=1, depth=1024)],

    # [FC(output=4096)]
]

def conv_layer_name(index,conv,depth_fn):
  if conv.stride == 1:
    layer_name = 'Conv2d_%s_%dx%d_%d'%(
        index, conv.kernel[0],conv.kernel[1],
        depth_fn(conv.depth))
  else:
    layer_name = 'Conv2d_%s_%dx%d_s%d_%d'%(
        index, conv.kernel[0],conv.kernel[1],conv.stride,
        depth_fn(conv.depth))
  return layer_name

def conv2d(net,conv,depth_fn,index):
  end_points = {}
  end_point = conv_layer_name(index,conv,depth_fn)
  net = slim.conv2d(net, depth_fn(conv.depth), conv.kernel,
                    stride=conv.stride,
                    normalizer_fn=slim.batch_norm,
                    scope=end_point)
  end_points[end_point] = net
  return net,end_points

def insert1x1_conv2d(net,conv,depth_fn,index):
  end_points = {}
  layer_name = 'Conv2d_%s_1x1_%d' % (index,conv.depth/2)
  intermediate_layer = slim.conv2d(
      net,
      depth_fn(conv.depth/2), [1, 1],
      padding='SAME',
      stride=1,
      scope=layer_name)
  end_points[layer_name]=intermediate_layer
  
  use_depthwise = False
  layer_name = conv_layer_name(index,conv,depth_fn)
  if use_depthwise:
    net = slim.separable_conv2d(
        intermediate_layer,
        None, conv.kernel,
        depth_multiplier=1,
        padding='SAME',
        stride=stride,
        scope=layer_name + '_depthwise')
    net = slim.conv2d(
        net,
        depth_fn(conv.depth), [1, 1],
        padding='SAME',
        stride=1,
        scope=layer_name)
  else:
    net = slim.conv2d(
        intermediate_layer,
        depth_fn(conv.depth), conv.kernel,
        padding='SAME',
        stride=conv.stride,
        scope=layer_name)
  end_points[layer_name]=net
  return net,end_points

def insert1x1_conv2d_rep(net,conv,depth_fn,index):
  end_points={}
  for i in range(0,conv.rep):
    rep_index='%s_%s'%(index,chr(i+ord('a')))
    net,endpoint=insert1x1_conv2d(net,conv,depth_fn,rep_index)
    end_points.update(endpoint)
  return net,end_points

def max_pool(net,conv,index):
  end_points={}
  end_point = 'MaxPool_%s_%dx%d' % (index,
    conv.kernel[0],conv.kernel[1])
  net=slim.max_pool2d(
    net, conv.kernel, 
    stride=conv.stride, padding='VALID',
    scope=end_point)
  end_points[end_point]=net
  return net,end_points

def fully_connect(net,conv,index):
  end_points={}
  end_point = 'FullyConnected_%s_%d' % (index,conv.output)
  net = slim.flatten(net)
  net = slim.fully_connected(net, conv.output, scope=end_point)
  end_points[end_point]=net
  return net,end_points

def googlenet_base(inputs,
                      final_endpoint='Conv2d_5b_3x3_1024',
                      min_depth=8,
                      depth_multiplier=1.0,
                      conv_defs=None,
                      output_stride=None,
                      scope=None):
  """GoogLeNet.

  Constructs a GoogLeNet network from inputs to the given final endpoint.

  Args:
    inputs: a tensor of shape [batch_size, height, width, channels].
    final_endpoint: specifies the endpoint to construct the network up to. It
      can be one of ['Conv2d_0a_7x7_s2_64', 'Conv2d_1a_3x3_192', 
      'Conv2d_2a_3x3_256','Conv2d_2b_3x3_512','Conv2d_3a_d_3x3_512',
      'Conv2d_3b_3x3_1024', 'Conv2d_4a_b_3x3_1024', 'Conv2d_4b_3x3_1024',
      'Conv2d_4c_3x3_s2_1024', 'Conv2d_5a_3x3_1024', 'Conv2d_5b_3x3_1024'].
    min_depth: Minimum depth value (number of channels) for all convolution ops.
      Enforced when depth_multiplier < 1, and not an active constraint when
      depth_multiplier >= 1.
    depth_multiplier: Float multiplier for the depth (number of channels)
      for all convolution ops. The value must be greater than zero. Typical
      usage will be to set this value in (0, 1) to reduce the number of
      parameters or computation cost of the model.
    conv_defs: A list of ConvDef namedtuples specifying the net architecture.
    output_stride: An integer that specifies the requested ratio of input to
      output spatial resolution. If not None, then we invoke atrous convolution
      if necessary to prevent the network from reducing the spatial resolution
      of the activation maps. Allowed values are 8 (accurate fully convolutional
      mode), 16 (fast fully convolutional mode), 32 (classification mode).
    scope: Optional variable_scope.

  Returns:
    tensor_out: output tensor corresponding to the final_endpoint.
    end_points: a set of activations for external use, for example summaries or
                losses.

  Raises:
    ValueError: if final_endpoint is not set to one of the predefined values,
                or depth_multiplier <= 0, or the target output_stride is not
                allowed.
  """
  depth_fn = lambda d: max(int(d * depth_multiplier), min_depth)
  end_points = {}

  # Used to find thinned depths for each layer.
  if depth_multiplier <= 0:
    raise ValueError('depth_multiplier is not greater than zero.')

  if conv_defs is None:
    conv_defs = _CONV_DEFS

  if output_stride is not None and output_stride not in [8, 16, 32]:
    raise ValueError('Only allowed output_stride values are 8, 16, 32.')

  with tf.variable_scope(scope, 'GoogLeNet', [inputs]):
    with slim.arg_scope([slim.conv2d, slim.separable_conv2d], padding='SAME'):
      # The current_stride variable keeps track of the output stride of the
      # activations, i.e., the running product of convolution strides up to the
      # current network layer. This allows us to invoke atrous convolution
      # whenever applying the next convolution would result in the activations
      # having output stride larger than the target output_stride.
      current_stride = 1

      # The atrous convolution rate parameter.
      rate = 1

      net = inputs
      for i, conv_def in enumerate(conv_defs):
        for ii, conv in enumerate(conv_def):
          index='%d%s'% (i,chr(ii+ord('a')))
          
          if isinstance(conv, Conv):
            net,end_point=conv2d(net,conv,depth_fn,index)
          elif isinstance(conv, Insert1x1Conv):
            net,end_point=insert1x1_conv2d(net,conv,depth_fn,index)
          elif isinstance(conv, Insert1x1ConvRep):
            net,end_point=insert1x1_conv2d_rep(net,conv,depth_fn,index) 
          elif isinstance(conv,Maxpool):
            net,end_point=max_pool(net,conv,index)
          elif isinstance(conv,FC):
            net,end_point=fully_connect(net,conv,index)
          else:
            raise ValueError('Unknown convolution type %s for layer %d'
                             % (conv.ltype, i))

          end_points.update(end_point)
          if final_endpoint in end_point:
            return net, end_points

  raise ValueError('Unknown final endpoint %s' % final_endpoint)


class YOLOGoogLeNetFeatureExtractor(yolo_meta_arch.YOLOFeatureExtractor):
  """YOLO Feature Extractor using GoogLeNet features."""

  def __init__(self,
               is_training,
               depth_multiplier,
               min_depth,
               pad_to_multiple,
               conv_hyperparams,
               batch_norm_trainable=True,
               reuse_weights=None):
    """MobileNetV1 Feature Extractor for YOLO Models.

    Args:
      is_training: whether the network is in training mode.
      depth_multiplier: float depth multiplier for feature extractor.
      min_depth: minimum feature extractor depth.
      pad_to_multiple: the nearest multiple to zero pad the input height and
        width dimensions to.
      conv_hyperparams: tf slim arg_scope for conv2d and separable_conv2d ops.
      batch_norm_trainable: Whether to update batch norm parameters during
        training or not. When training with a small batch size
        (e.g. 1), it is desirable to disable batch norm update and use
        pretrained batch norm params.
      reuse_weights: Whether to reuse variables. Default is None.
    """
    super(YOLOGoogLeNetFeatureExtractor, self).__init__(
        is_training, depth_multiplier, min_depth, pad_to_multiple,
        conv_hyperparams, batch_norm_trainable, reuse_weights)

  def preprocess(self, resized_inputs):
    """YOLO preprocessing.

    Maps pixel values to the range [-1, 1].

    Args:
      resized_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.

    Returns:
      preprocessed_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.
    """
    return (2.0 / 255.0) * resized_inputs - 1.0

  def extract_features(self, preprocessed_inputs):
    """Extract features from preprocessed inputs.

    Args:
      preprocessed_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.

    Returns:
      feature_maps: a list of tensors where the ith tensor has shape
        [batch, height_i, width_i, depth_i]
    """
    preprocessed_inputs.get_shape().assert_has_rank(4)
    shape_assert = tf.Assert(
        tf.logical_and(tf.greater_equal(tf.shape(preprocessed_inputs)[1], 33),
                       tf.greater_equal(tf.shape(preprocessed_inputs)[2], 33)),
        ['image size must at least be 33 in both height and width.'])

    feature_map_layout = {
        'from_layer': ['Conv2d_5b_3x3_1024'],
        'layer_depth': [-1],
    }

    with tf.control_dependencies([shape_assert]):
      with slim.arg_scope(self._conv_hyperparams):
        with slim.arg_scope([slim.batch_norm], fused=False):
          with tf.variable_scope('GoogLeNet',
                                 reuse=self._reuse_weights) as scope:
            _, image_features = googlenet_base(
                ops.pad_to_multiple(preprocessed_inputs, self._pad_to_multiple),
                final_endpoint='Conv2d_5b_3x3_1024',
                min_depth=self._min_depth,
                depth_multiplier=self._depth_multiplier,
                scope=scope)
            feature_maps = feature_map_generators.multi_resolution_feature_maps(
                feature_map_layout=feature_map_layout,
                depth_multiplier=self._depth_multiplier,
                min_depth=self._min_depth,
                insert_1x1_conv=True,
                image_features=image_features)

    return feature_maps.values()

  def init_net_base(self, preprocessed_inputs):
    """Init network for feature extraction.

    Args:
      preprocessed_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.

    Returns:
      end_points: a dict of tensors where the ith tensor has shape
        [batch, height_i, width_i, depth_i]
    """
    preprocessed_inputs.get_shape().assert_has_rank(4)
    shape_assert = tf.Assert(
        tf.logical_and(tf.greater_equal(tf.shape(preprocessed_inputs)[1], 33),
                       tf.greater_equal(tf.shape(preprocessed_inputs)[2], 33)),
        ['image size must at least be 33 in both height and width.'])

    with tf.control_dependencies([shape_assert]):
      with slim.arg_scope(self._conv_hyperparams):
        with slim.arg_scope([slim.batch_norm], fused=False):
          with tf.variable_scope('GoogLeNet',
                                 reuse=self._reuse_weights) as scope:
            _, end_points = googlenet_base(
                ops.pad_to_multiple(preprocessed_inputs, self._pad_to_multiple),
                final_endpoint='Conv2d_5b_3x3_1024',
                min_depth=self._min_depth,
                depth_multiplier=self._depth_multiplier,
                scope=scope)

    return end_points

#   def bn_act_conv_drp(current, num_outputs, kernel_size, scope='block'):
#     current = slim.batch_norm(current, scope=scope + '_bn')
#     current = tf.nn.relu(current)
#     current = slim.conv2d(current, num_outputs, kernel_size, scope=scope + '_conv')
#     current = slim.dropout(current, scope=scope + '_dropout')
#                 net=slim.max_pool2d(net, [3, 3], stride=2, padding='same',
# scope='MaxPool_0b_3x3')
#     return current

