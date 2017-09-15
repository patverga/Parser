#!/usr/bin/env python
# -*- coding: UTF-8 -*-
 
# Copyright 2016 Timothy Dozat
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from lib import linalg
from lib.etc.tarjan import Tarjan
from lib.models import rnn
from configurable import Configurable
from vocab import Vocab

import scipy.linalg

def layer_norm(inputs, reuse, epsilon=1e-6):
  """Applies layer normalization.

  Args:
    inputs: A tensor with 2 or more dimensions, where the first dimension has
      `batch_size`.
    epsilon: A floating number. A very small number for preventing ZeroDivision Error.

  Returns:
    A tensor with the same shape and data dtype as `inputs`.
  """
  with tf.variable_scope("layer_norm", reuse=reuse):
    inputs_shape = inputs.get_shape()
    params_shape = inputs_shape[-1:]

    mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
    beta = tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer())
    gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())
    normalized = (inputs - mean) * tf.rsqrt(variance + epsilon)
    outputs = gamma * normalized + beta
  return outputs


def add_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4):
  """Adds a bunch of sinusoids of different frequencies to a Tensor.
  Each channel of the input Tensor is incremented by a sinusoid of a different
  frequency and phase.
  This allows attention to learn to use absolute and relative positions.
  Timing signals should be added to some precursors of both the query and the
  memory inputs to attention.
  The use of relative position is possible because sin(x+y) and cos(x+y) can be
  experessed in terms of y, sin(x) and cos(x).
  In particular, we use a geometric sequence of timescales starting with
  min_timescale and ending with max_timescale.  The number of different
  timescales is equal to channels / 2. For each timescale, we
  generate the two sinusoidal signals sin(timestep/timescale) and
  cos(timestep/timescale).  All of these sinusoids are concatenated in
  the channels dimension.
  Args:
    x: a Tensor with shape [batch, length, channels]
    min_timescale: a float
    max_timescale: a float
  Returns:
    a Tensor the same shape as x.
  """
  length = tf.shape(x)[1]
  channels = tf.shape(x)[2]
  position = tf.to_float(tf.range(length))
  num_timescales = channels // 2
  log_timescale_increment = (
      np.log(float(max_timescale) / float(min_timescale)) /
      (tf.to_float(num_timescales) - 1))
  inv_timescales = min_timescale * tf.exp(
      tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
  scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
  signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
  signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
  signal = tf.reshape(signal, [1, length, channels])
  return x + signal


def attention_bias_ignore_padding(lengths):
  """Create an bias tensor to be added to attention logits.
  Args:
    memory_padding: a float `Tensor` with shape [batch, memory_length].
  Returns:
    a `Tensor` with shape [batch, 1, 1, memory_length].
  """
  mask = tf.sequence_mask(lengths, tf.reduce_max(lengths)+1)
  memory_padding = tf.cast(tf.logical_not(mask), tf.float32)
  ret = memory_padding * -1e9
  return tf.expand_dims(tf.expand_dims(ret, axis=1), axis=1)


def split_last_dimension(x, n):
  """Reshape x so that the last dimension becomes two dimensions.
  The first of these two dimensions is n.
  Args:
    x: a Tensor with shape [..., m]
    n: an integer.
  Returns:
    a Tensor with shape [..., n, m/n]
  """
  old_shape = x.get_shape().dims
  last = old_shape[-1]
  new_shape = old_shape[:-1] + [n] + [last // n if last else None]
  ret = tf.reshape(x, tf.concat([tf.shape(x)[:-1], [n, -1]], 0))
  ret.set_shape(new_shape)
  return ret


def combine_last_two_dimensions(x):
  """Reshape x so that the last two dimension become one.
  Args:
    x: a Tensor with shape [..., a, b]
  Returns:
    a Tensor with shape [..., ab]
  """
  old_shape = x.get_shape().dims
  a, b = old_shape[-2:]
  new_shape = old_shape[:-2] + [a * b if a and b else None]
  ret = tf.reshape(x, tf.concat([tf.shape(x)[:-2], [-1]], 0))
  ret.set_shape(new_shape)
  return ret


def split_heads(x, num_heads):
  """Split channels (dimension 3) into multiple heads (becomes dimension 1).
  Args:
    x: a Tensor with shape [batch, length, channels]
    num_heads: an integer
  Returns:
    a Tensor with shape [batch, num_heads, length, channels / num_heads]
  """
  return tf.transpose(split_last_dimension(x, num_heads), [0, 2, 1, 3])


def combine_heads(x):
  """Inverse of split_heads.
  Args:
    x: a Tensor with shape [batch, num_heads, length, channels / num_heads]
  Returns:
    a Tensor with shape [batch, length, channels]
  """
  return combine_last_two_dimensions(tf.transpose(x, [0, 2, 1, 3]))


def dot_product_attention(q, k, v,
                          bias,
                          dropout_rate=1.0,
                          name=None):
  """dot-product attention.
  Args:
    q: a Tensor with shape [batch, heads, length_q, depth_k]
    k: a Tensor with shape [batch, heads, length_kv, depth_k]
    v: a Tensor with shape [batch, heads, length_kv, depth_v]
    bias: bias Tensor (see attention_bias())
    dropout_rate: a floating point number
    name: an optional string
  Returns:
    A Tensor.
  """
  with tf.variable_scope(name, default_name="dot_product_attention", values=[q, k, v]):
    # [batch, num_heads, query_length, memory_length]
    logits = tf.matmul(q, k, transpose_b=True)
    if bias is not None:
      logits += bias
    weights = tf.nn.softmax(logits, name="attention_weights")
    # dropping out the attention links for each of the heads
    weights = tf.nn.dropout(weights, dropout_rate)
    return tf.matmul(weights, v)


def compute_qkv(antecedent, total_key_depth, total_value_depth):
  """Computes query, key and value.
  Args:
    total_key_depth: an integer
    total_value_depth: and integer
  Returns:
    q, k, v : [batch, length, depth] tensors
  """
  params = tf.get_variable("qkv_transform", [1, 1, total_key_depth, 2*total_key_depth + total_value_depth])
  antecedent = tf.expand_dims(antecedent, 1)
  qkv_combined = tf.nn.conv2d(antecedent, params, [1, 1, 1, 1], "SAME")
  qkv_combined = tf.squeeze(qkv_combined, 1)
  q, k, v = tf.split(qkv_combined, [total_key_depth, total_key_depth, total_value_depth], axis=2)
  return q, k, v


def multihead_attention(antecedent,
                        bias,
                        total_key_depth,
                        total_value_depth,
                        output_depth,
                        num_heads,
                        dropout_rate,
                        name=None):
  """Multihead scaled-dot-product attention with input/output transformations.
  Args:
    bias: bias Tensor (see attention_bias())
    total_key_depth: an integer
    total_value_depth: an integer
    output_depth: an integer
    num_heads: an integer dividing total_key_depth and total_value_depth
    dropout_rate: a floating point number
    name: an optional string
  Returns:
    A Tensor.
  Raises:
    ValueError: if the key depth or value depth are not divisible by the
      number of attention heads.
  """
  if total_key_depth % num_heads != 0:
    raise ValueError("Key depth (%d) must be divisible by the number of "
                     "attention heads (%d)." % (total_key_depth, num_heads))
  if total_value_depth % num_heads != 0:
    raise ValueError("Value depth (%d) must be divisible by the number of "
                     "attention heads (%d)." % (total_value_depth, num_heads))
  with tf.variable_scope(name, default_name="multihead_attention", values=[antecedent]):
    q, k, v = compute_qkv(antecedent, total_key_depth, total_value_depth)
    q = split_heads(q, num_heads)
    k = split_heads(k, num_heads)
    v = split_heads(v, num_heads)
    key_depth_per_head = total_key_depth // num_heads
    q *= key_depth_per_head**-0.5
    x = dot_product_attention(q, k, v, bias, dropout_rate)
    x = combine_heads(x)
    params = tf.get_variable("final_proj", [1, 1, total_key_depth, output_depth])
    x = tf.expand_dims(x, 1)
    x = tf.nn.conv2d(x, params, [1, 1, 1, 1], "SAME")
    x = tf.squeeze(x, 1)
    return x


def conv_hidden_relu(inputs,
                     hidden_size,
                     output_size,
                     dropout,
                     nonlinearity):
  """Hidden layer with RELU activation followed by linear projection."""
  with tf.variable_scope("conv_hidden_relu", [inputs]):
    inputs = tf.expand_dims(inputs, 1)
    in_size = inputs.get_shape().as_list()[-1]
    kernel = 3
    params1 = tf.get_variable("ff1", [1, 1, in_size, hidden_size])
    params2 = tf.get_variable("ff2", [1, kernel, hidden_size, hidden_size])
    params3 = tf.get_variable("ff3", [1, 1, hidden_size, output_size])
    h = tf.nn.conv2d(inputs, params1, [1, 1, 1, 1], "SAME")
    h = nonlinearity(h)
    h = tf.nn.dropout(h, dropout)
    h = tf.nn.conv2d(h, params2, [1, 1, 1, 1], "SAME")
    h = nonlinearity(h)
    h = tf.nn.dropout(h, dropout)
    ret = tf.nn.conv2d(h, params3, [1, 1, 1, 1], "SAME")
    ret = tf.squeeze(ret, 1)
    return ret


#***************************************************************
class NN(Configurable):
  """"""
  
  ZERO = tf.convert_to_tensor(0.)
  ONE = tf.convert_to_tensor(1.)
  PUNCT = set(['``', "''", ':', ',', '.', 'PU', 'PUNCT'])
  
  #=============================================================
  def __init__(self, *args, **kwargs):
    """"""
    
    global_step = kwargs.pop('global_step', None)
    super(NN, self).__init__(*args, **kwargs)
    
    if global_step is not None:
      self._global_sigmoid = 1-tf.nn.sigmoid(3*(2*global_step/(self.train_iters-1)-1))
    else:
      self._global_sigmoid = 1
    
    self.tokens_to_keep3D = None
    self.sequence_lengths = None
    self.n_tokens = None
    self.moving_params = None
    return
  
  #=============================================================
  def embed_concat(self, word_inputs, tag_inputs=None, rel_inputs=None):
    """"""
    
    if self.moving_params is None:
      word_keep_prob = self.word_keep_prob
      tag_keep_prob = self.tag_keep_prob
      rel_keep_prob = self.rel_keep_prob
      noise_shape = tf.stack([tf.shape(word_inputs)[0], tf.shape(word_inputs)[1], 1])
      
      if word_keep_prob < 1:
        word_mask = tf.nn.dropout(tf.ones(noise_shape), word_keep_prob)*word_keep_prob
      else:
        word_mask = 1
      if tag_inputs is not None and tag_keep_prob < 1:
        tag_mask = tf.nn.dropout(tf.ones(noise_shape), tag_keep_prob)*tag_keep_prob
      else:
        tag_mask = 1
      if rel_inputs is not None and rel_keep_prob < 1:
        rel_mask = tf.nn.dropout(tf.ones(noise_shape), rel_keep_prob)*rel_keep_prob
      else:
        rel_mask = 1
      
      word_embed_size = word_inputs.get_shape().as_list()[-1]
      tag_embed_size = 0 if tag_inputs is None else tag_inputs.get_shape().as_list()[-1]
      rel_embed_size = 0 if rel_inputs is None else rel_inputs.get_shape().as_list()[-1]
      total_size = word_embed_size + tag_embed_size + rel_embed_size
      if word_embed_size == tag_embed_size:
        total_size += word_embed_size
      dropped_sizes = word_mask * word_embed_size + tag_mask * tag_embed_size + rel_mask * rel_embed_size
      if word_embed_size == tag_embed_size:
        dropped_sizes += word_mask * tag_mask * word_embed_size
      scale_factor = total_size / (dropped_sizes + self.epsilon)
      
      word_inputs *= word_mask * scale_factor
      if tag_inputs is not None:
        tag_inputs *= tag_mask * scale_factor
      if rel_inputs is not None:
        rel_inputs *= rel_mask * scale_factor
    else:
      word_embed_size = word_inputs.get_shape().as_list()[-1]
      tag_embed_size = 0 if tag_inputs is None else tag_inputs.get_shape().as_list()[-1]
      rel_embed_size = 0 if rel_inputs is None else rel_inputs.get_shape().as_list()[-1]
    
    return tf.concat(axis=2, values=filter(lambda x: x is not None, [word_inputs, tag_inputs, rel_inputs]))
  
  #=============================================================
  def RNN(self, inputs):
    """"""
    
    input_size = inputs.get_shape().as_list()[-1]
    cell = self.recur_cell(self._config, input_size=input_size, moving_params=self.moving_params)
    lengths = tf.reshape(tf.to_int64(self.sequence_lengths), [-1])
    
    if self.moving_params is None:
      ff_keep_prob = self.ff_keep_prob
      recur_keep_prob = self.recur_keep_prob
    else:
      ff_keep_prob = 1
      recur_keep_prob = 1
    
    if self.recur_bidir:
      top_recur, fw_recur, bw_recur = rnn.dynamic_bidirectional_rnn(cell, cell, inputs,
                                                                    lengths,
                                                                    ff_keep_prob=ff_keep_prob,
                                                                    recur_keep_prob=recur_keep_prob,
                                                                    dtype=tf.float32)
      fw_cell, fw_out = tf.split(axis=1, num_or_size_splits=2, value=fw_recur)
      bw_cell, bw_out = tf.split(axis=1, num_or_size_splits=2, value=bw_recur)
      end_recur = tf.concat(axis=1, values=[fw_out, bw_out])
      top_recur.set_shape([tf.Dimension(None), tf.Dimension(None), tf.Dimension(2*self.recur_size)])
    else:
      top_recur, end_recur = rnn.dynamic_rnn(cell, inputs,
                                             lengths,
                                             ff_keep_prob=ff_keep_prob,
                                             recur_keep_prob=recur_keep_prob,
                                             dtype=tf.float32)
      top_recur.set_shape([tf.Dimension(None), tf.Dimension(None), tf.Dimension(self.recur_size)])
    return top_recur, end_recur

  # =============================================================
  def CNN(self, inputs, kernel, output_size, dropout_keep_rate, nonlinearity):
    """"""
    input_size = inputs.get_shape().as_list()[-1]

    initializer = tf.contrib.layers.xavier_initializer()
    # mat = linalg.orthonormal_initializer(input_size, output_size)
    # initializer = tf.constant_initializer(mat)

    if self.moving_params is not None:
      dropout_keep_rate = 1.0

    params = tf.get_variable('CNN', [1, kernel, input_size, output_size], initializer=initializer)
    inputs = tf.expand_dims(inputs, 1)
    conv_out = tf.nn.conv2d(inputs, params, [1, 1, 1, 1], 'SAME')
    conv_out = tf.squeeze(conv_out, 1)
    conv_out = nonlinearity(conv_out)
    conv_out = tf.nn.dropout(conv_out, dropout_keep_rate)
    return conv_out

  # =============================================================
  def transformer(self, inputs, hidden_size, num_heads, attn_dropout, relu_dropout, prepost_dropout, relu_hidden_size,
                  nonlinearity, reuse):
    """"""
    # input_size = inputs.get_shape().as_list()[-1]
    lengths = tf.reshape(tf.to_int64(self.sequence_lengths), [-1])
    mask = attention_bias_ignore_padding(lengths)

    # mat = linalg.orthonormal_initializer(input_size, output_size)
    # initializer = tf.constant_initializer(mat)
    if self.moving_params is not None:
      attn_dropout = 1.0
      relu_dropout = 1.0
      prepost_dropout = 1.0

    with tf.variable_scope("self_attention"):
      x = layer_norm(inputs, reuse)
      y = multihead_attention(x, mask, hidden_size, hidden_size, hidden_size, num_heads, attn_dropout)
      x = tf.add(x, tf.nn.dropout(y, prepost_dropout))

    with tf.variable_scope("ffnn"):
      x = layer_norm(x, reuse)
      y = conv_hidden_relu(x, relu_hidden_size, hidden_size, relu_dropout, nonlinearity)
      x = tf.add(x, tf.nn.dropout(y, prepost_dropout))

    return x
  
  #=============================================================
  def soft_attn(self, top_recur):
    """"""
    
    reuse = (self.moving_params is not None) or None
    
    input_size = top_recur.get_shape().as_list()[-1]
    with tf.variable_scope('MLP', reuse=reuse):
      head_mlp, dep_mlp = self.MLP(top_recur, self.info_mlp_size,
                                   func=self.info_func,
                                   keep_prob=self.info_keep_prob,
                                   n_splits=2)
    with tf.variable_scope('Arcs', reuse=reuse):
      arc_logits = self.bilinear_classifier(dep_mlp, head_mlp, keep_prob=self.info_keep_prob)
      arc_prob = self.softmax(arc_logits)
      head_lin = tf.matmul(arc_prob, top_recur)
      top_recur = tf.concat(axis=2, values=[top_recur, head_lin])
    top_recur.set_shape([tf.Dimension(None), tf.Dimension(None), tf.Dimension(4*self.recur_size)])
    return top_recur

  #=============================================================
  def linear(self, inputs, output_size, n_splits=1, add_bias=False):
    """"""
    
    n_dims = len(inputs.get_shape().as_list())
    batch_size = tf.shape(inputs)[0]
    bucket_size = tf.shape(inputs)[1]
    input_size = inputs.get_shape().as_list()[-1]
    output_shape = tf.stack([batch_size] + [bucket_size]*(n_dims-2) + [output_size])
    shape_to_set = [tf.Dimension(None)]*(n_dims-1) + [tf.Dimension(output_size)]
    
    if self.moving_params is None:
      keep_prob = self.info_keep_prob
    else:
      keep_prob = 1
    
    if keep_prob < 1:
      noise_shape = tf.stack([batch_size] + [1]*(n_dims-2) + [input_size])
      inputs = tf.nn.dropout(inputs, keep_prob, noise_shape=noise_shape)

    lin = linalg.linear(inputs,
                        output_size,
                        n_splits=n_splits,
                        add_bias=add_bias,
                        moving_params=self.moving_params)
    if n_splits == 1:
      lin = [lin]
    for i, split in enumerate(lin):
      split.set_shape(shape_to_set)
    if n_splits == 1:
      return lin[0]
    else:
      return lin

  #=============================================================
  def softmax(self, inputs):
    """"""
    
    input_shape = tf.shape(inputs)
    batch_size = input_shape[0]
    bucket_size = input_shape[1]
    input_size = input_shape[2]
    inputs = tf.reshape(inputs, tf.stack([-1, input_size]))
    probs = tf.nn.softmax(inputs)
    probs = tf.reshape(probs, tf.stack([batch_size, bucket_size, input_size]))
    return probs
  
  #=============================================================
  def MLP(self, inputs, output_size, func=None, keep_prob=None, n_splits=1):
    """"""
    
    n_dims = len(inputs.get_shape().as_list())
    batch_size = tf.shape(inputs)[0]
    bucket_size = tf.shape(inputs)[1]
    input_size = inputs.get_shape().as_list()[-1]
    output_shape = tf.stack([batch_size] + [bucket_size]*(n_dims-2) + [output_size])
    shape_to_set = [tf.Dimension(None)]*(n_dims-1) + [tf.Dimension(output_size)]
    if func is None:
      func = self.mlp_func
    
    if self.moving_params is None:
      if keep_prob is None:
        keep_prob = self.mlp_keep_prob
    else:
      keep_prob = 1
    if keep_prob < 1:
      noise_shape = tf.stack([batch_size] + [1]*(n_dims-2) + [input_size])
      inputs = tf.nn.dropout(inputs, keep_prob, noise_shape=noise_shape)
    
    linear = linalg.linear(inputs,
                        output_size,
                        n_splits=n_splits * (1+(func.__name__ in ('gated_tanh', 'gated_identity'))),
                        add_bias=True,
                        moving_params=self.moving_params)
    if func.__name__ in ('gated_tanh', 'gated_identity'):
      linear = [tf.concat(axis=n_dims-1, values=[lin1, lin2]) for lin1, lin2 in zip(linear[:len(linear)//2], linear[len(linear)//2:])]
    if n_splits == 1:
      linear = [linear]
    for i, split in enumerate(linear):
      split = func(split)
      split.set_shape(shape_to_set)
      linear[i] = split
    if n_splits == 1:
      return linear[0]
    else:
      return linear
  
  #=============================================================
  def double_MLP(self, inputs, n_splits=1):
    """"""
    
    batch_size = tf.shape(inputs)[0]
    bucket_size = tf.shape(inputs)[1]
    input_size = inputs.get_shape().as_list()[-1]
    output_size = self.attn_mlp_size
    output_shape = tf.stack([batch_size, bucket_size, bucket_size, output_size])
    shape_to_set = [tf.Dimension(None), tf.Dimension(None), tf.Dimension(None), tf.Dimension(output_size)]
    
    if self.moving_params is None:
      keep_prob = self.mlp_keep_prob
    else:
      keep_prob = 1
    if isinstance(keep_prob, tf.Tensor) or keep_prob < 1:
      noise_shape = tf.stack([batch_size, 1, input_size])
      inputs = tf.nn.dropout(inputs, keep_prob, noise_shape=noise_shape)
    
    lin1, lin2 = linalg.linear(inputs,
                               output_size*n_splits,
                               n_splits=2,
                               add_bias=True,
                               moving_params=self.moving_params)
    lin1 = tf.reshape(tf.transpose(lin1, [0, 2, 1]), tf.stack([-1, bucket_size, 1]))
    lin2 = tf.reshape(tf.transpose(lin2, [0, 2, 1]), tf.stack([-1, 1, bucket_size]))
    lin = lin1 + lin2
    lin = tf.reshape(lin, tf.stack([batch_size, n_splits*output_size, bucket_size, bucket_size]))
    lin = tf.transpose(lin, [0,2,3,1])
    top_mlps = tf.split(axis=3, num_or_size_splits=n_splits, value=self.mlp_func(lin))
    for top_mlp in top_mlps:
      top_mlp.set_shape(shape_to_set)
    if n_splits == 1:
      return top_mlps[0]
    else:
      return top_mlps
  
  #=============================================================
  def linear_classifier(self, inputs, n_classes, add_bias=True, keep_prob=None):
    """"""
    
    n_dims = len(inputs.get_shape().as_list())
    batch_size = tf.shape(inputs)[0]
    bucket_size = tf.shape(inputs)[1]
    input_size = inputs.get_shape().as_list()[-1]
    output_size = n_classes
    output_shape = tf.stack([batch_size] + [bucket_size]*(n_dims-2) + [output_size])
    
    if self.moving_params is None:
      if keep_prob is None:
        keep_prob = self.mlp_keep_prob
    else:
      keep_prob = 1
    if isinstance(keep_prob, tf.Tensor) or keep_prob < 1:
      noise_shape = tf.stack([batch_size] + [1]*(n_dims-2) +[input_size])
      inputs = tf.nn.dropout(inputs, keep_prob, noise_shape=noise_shape)
    
    inputs = tf.reshape(inputs, [-1, input_size])
    output = linalg.linear(inputs,
                    output_size,
                    add_bias=add_bias,
                    initializer=tf.zeros_initializer(),
                    moving_params=self.moving_params)
    output = tf.reshape(output, output_shape)
    output.set_shape([tf.Dimension(None)]*(n_dims-1) + [tf.Dimension(output_size)])
    return output
  
  #=============================================================
  def bilinear_classifier(self, inputs1, inputs2, add_bias1=True, add_bias2=False, keep_prob=None):
    """"""
    
    input_shape = tf.shape(inputs1)
    batch_size = input_shape[0]
    bucket_size = input_shape[1]
    input_size = inputs1.get_shape().as_list()[-1]
    
    if self.moving_params is None:
      if keep_prob is None:
        keep_prob = self.mlp_keep_prob
    else:
      keep_prob = 1
    if isinstance(keep_prob, tf.Tensor) or keep_prob < 1:
      noise_shape = tf.stack([batch_size, 1, input_size])
      # Experimental
      #inputs1 = tf.nn.dropout(inputs1, keep_prob if add_bias2 else tf.sqrt(keep_prob), noise_shape=noise_shape)
      #inputs2 = tf.nn.dropout(inputs2, keep_prob if add_bias1 else tf.sqrt(keep_prob), noise_shape=noise_shape)
      inputs1 = tf.nn.dropout(inputs1, keep_prob, noise_shape=noise_shape)
      inputs2 = tf.nn.dropout(inputs2, keep_prob, noise_shape=noise_shape)
    
    bilin = linalg.bilinear(inputs1, inputs2, 1,
                            add_bias1=add_bias1,
                            add_bias2=add_bias2,
                            initializer=tf.zeros_initializer(),
                            moving_params=self.moving_params)
    output = tf.squeeze(bilin)
    return output
  
  #=============================================================
  def diagonal_bilinear_classifier(self, inputs1, inputs2, add_bias1=True, add_bias2=False):
    """"""
    
    input_shape = tf.shape(inputs1)
    batch_size = input_shape[0]
    bucket_size = input_shape[1]
    input_size = inputs1.get_shape().as_list()[-1]
    shape_to_set = tf.stack([batch_size, bucket_size, input_size+1])
    
    if self.moving_params is None:
      keep_prob = self.mlp_keep_prob
    else:
      keep_prob = 1
    if isinstance(keep_prob, tf.Tensor) or keep_prob < 1:
      noise_shape = tf.stack([batch_size, 1, input_size])
      inputs1 = tf.nn.dropout(inputs1, tf.sqrt(keep_prob), noise_shape=noise_shape)
      inputs2 = tf.nn.dropout(inputs2, tf.sqrt(keep_prob), noise_shape=noise_shape)
    
    bilin = linalg.diagonal_bilinear(inputs1, inputs2, 1,
                                     add_bias1=add_bias1,
                                     add_bias2=add_bias2,
                                     initializer=tf.zeros_initializer(),
                                     moving_params=self.moving_params)
    output = tf.squeeze(bilin)
    return output
  
  #=============================================================
  def conditional_linear_classifier(self, inputs, n_classes, probs, add_bias=True):
    """"""
    
    input_shape = tf.shape(inputs)
    batch_size = input_shape[0]
    bucket_size = input_shape[1]
    input_size = inputs.get_shape().as_list()[-1]
    
    if len(probs.get_shape().as_list()) == 2:
      probs = tf.to_float(tf.one_hot(tf.to_int64(probs), bucket_size, 1, 0))
    else:
      probs = tf.stop_gradient(probs)
    
    if self.moving_params is None:
      keep_prob = self.mlp_keep_prob
    else:
      keep_prob = 1
    if isinstance(keep_prob, tf.Tensor) or keep_prob < 1:
      noise_shape = tf.stack([batch_size, 1, 1, input_size])
      inputs = tf.nn.dropout(inputs, keep_prob, noise_shape=noise_shape)
    
    lin = linalg.linear(inputs,
                        n_classes,
                        add_bias=add_bias,
                        initializer=tf.zeros_initializer(),
                        moving_params=self.moving_params)
    weighted_lin = tf.matmul(lin, tf.expand_dims(probs, 3), adjoint_a=True)
    
    return weighted_lin, lin
  
  #=============================================================
  def conditional_diagonal_bilinear_classifier(self, inputs1, inputs2, n_classes, probs, add_bias1=True, add_bias2=True):
    """"""
    
    input_shape = tf.shape(inputs1)
    batch_size = input_shape[0]
    bucket_size = input_shape[1]
    input_size = inputs1.get_shape().as_list()[-1]
    input_shape_to_set = [tf.Dimension(None), tf.Dimension(None), input_size+1]
    output_shape = tf.stack([batch_size, bucket_size, n_classes, bucket_size])
    if len(probs.get_shape().as_list()) == 2:
      probs = tf.to_float(tf.one_hot(tf.to_int64(probs), bucket_size, 1, 0))
    else:
      probs = tf.stop_gradient(probs)
    
    if self.moving_params is None:
      keep_prob = self.mlp_keep_prob
    else:
      keep_prob = 1
    if isinstance(keep_prob, tf.Tensor) or keep_prob < 1:
      noise_shape = tf.stack([batch_size, 1, input_size])
      inputs1 = tf.nn.dropout(inputs1, tf.sqrt(keep_prob), noise_shape=noise_shape)
      inputs2 = tf.nn.dropout(inputs2, tf.sqrt(keep_prob), noise_shape=noise_shape)
    
    inputs1 = tf.concat(axis=2, values=[inputs1, tf.ones(tf.stack([batch_size, bucket_size, 1]))])
    inputs1.set_shape(input_shape_to_set)
    inputs2 = tf.concat(axis=2, values=[inputs2, tf.ones(tf.stack([batch_size, bucket_size, 1]))])
    inputs2.set_shape(input_shape_to_set)
    
    bilin = linalg.diagonal_bilinear(inputs1, inputs2,
                                     n_classes,
                                     add_bias1=add_bias1,
                                     add_bias2=add_bias2,
                                     initializer=tf.zeros_initializer(),
                                     moving_params=self.moving_params)
    weighted_bilin = tf.matmul(bilin, tf.expand_dims(probs, 3))
    
    return weighted_bilin, bilin
  
  #=============================================================
  def conditional_bilinear_classifier(self, inputs1, inputs2, n_classes, probs, add_bias1=True, add_bias2=True):
    """"""
    
    input_shape = tf.shape(inputs1)
    batch_size = input_shape[0]
    bucket_size = input_shape[1]
    input_size = inputs1.get_shape().as_list()[-1]
    input_shape_to_set = [tf.Dimension(None), tf.Dimension(None), input_size+1]
    output_shape = tf.stack([batch_size, bucket_size, n_classes, bucket_size])
    if len(probs.get_shape().as_list()) == 2:
      probs = tf.to_float(tf.one_hot(tf.to_int64(probs), bucket_size, 1, 0))
    else:
      probs = tf.stop_gradient(probs)
    
    if self.moving_params is None:
      keep_prob = self.mlp_keep_prob
    else:
      keep_prob = 1
    if isinstance(keep_prob, tf.Tensor) or keep_prob < 1:
      noise_shape = tf.stack([batch_size, 1, input_size])
      inputs1 = tf.nn.dropout(inputs1, keep_prob, noise_shape=noise_shape)
      inputs2 = tf.nn.dropout(inputs2, keep_prob, noise_shape=noise_shape)
    
    inputs1 = tf.concat(axis=2, values=[inputs1, tf.ones(tf.stack([batch_size, bucket_size, 1]))])
    inputs1.set_shape(input_shape_to_set)
    inputs2 = tf.concat(axis=2, values=[inputs2, tf.ones(tf.stack([batch_size, bucket_size, 1]))])
    inputs2.set_shape(input_shape_to_set)
    
    bilin = linalg.bilinear(inputs1, inputs2,
                     n_classes,
                     add_bias1=add_bias1,
                     add_bias2=add_bias2,
                     initializer=tf.zeros_initializer(),
                     moving_params=self.moving_params)
    weighted_bilin = tf.matmul(bilin, tf.expand_dims(probs, 3))
    
    return weighted_bilin, bilin
  
  #=============================================================
  def output(self, logits3D, targets3D):
    """"""
    
    original_shape = tf.shape(logits3D)
    batch_size = original_shape[0]
    bucket_size = original_shape[1]
    flat_shape = tf.stack([batch_size, bucket_size])
    
    logits2D = tf.reshape(logits3D, tf.stack([batch_size*bucket_size, -1]))
    targets1D = tf.reshape(targets3D, [-1])
    tokens_to_keep1D = tf.reshape(self.tokens_to_keep3D, [-1])
    
    predictions1D = tf.to_int32(tf.argmax(logits2D, 1))
    probabilities2D = tf.nn.softmax(logits2D)
    cross_entropy1D = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits2D, labels=targets1D)
    
    correct1D = tf.to_float(tf.equal(predictions1D, targets1D))
    n_correct = tf.reduce_sum(correct1D * tokens_to_keep1D)
    accuracy = n_correct / self.n_tokens
    loss = tf.reduce_sum(cross_entropy1D * tokens_to_keep1D) / self.n_tokens
    
    output = {
      'probabilities': tf.reshape(probabilities2D, original_shape),
      'predictions': tf.reshape(predictions1D, flat_shape),
      'tokens': tokens_to_keep1D,
      'correct': correct1D * tokens_to_keep1D,
      'n_correct': n_correct,
      'n_tokens': self.n_tokens,
      'accuracy': accuracy,
      'loss': loss
    }
    
    return output

  # =============================================================
  def output_svd(self, logits3D, targets3D):
    """"""

    original_shape = tf.shape(logits3D)
    batch_size = original_shape[0]
    bucket_size = original_shape[1]
    flat_shape = tf.stack([batch_size, bucket_size])

    # flatten to [B*N, N]
    logits2D = tf.reshape(logits3D, tf.stack([batch_size * bucket_size, -1]))

    targets1D = tf.reshape(targets3D, [-1])
    tokens_to_keep1D = tf.reshape(self.tokens_to_keep3D, [-1])

    # svd loss
    maxes = tf.expand_dims(tf.reduce_max(logits2D, axis=1), 1)
    maxes_tiled = tf.tile(maxes, [1, bucket_size])
    adj_flat = tf.cast(tf.equal(logits2D, maxes_tiled), tf.float32)
    adj = tf.reshape(adj_flat, [batch_size, bucket_size, bucket_size])
    adj = tf.matrix_set_diag(adj, tf.zeros([batch_size, bucket_size]))

    degrees = tf.reduce_sum(adj, axis=1)
    laplacian = tf.matrix_set_diag(-adj, degrees)

    dtype = laplacian.dtype
    _, s, _ = tf.py_func(np.linalg.svd, [laplacian, False, True], [dtype, dtype, dtype])
    # s, _, _ = tf.svd(laplacian)



    l_trace = tf.reduce_sum(degrees, axis=1)
    l_rank = tf.reduce_sum(tf.cast(tf.greater(s, 1e-15), tf.float32), axis=1)

    svd_loss = tf.maximum(0.5 * l_trace - (l_rank + 1), tf.constant(0.0))
    svd_loss_masked = self.tokens_to_keep3D * svd_loss
    svd_loss_avg = tf.reduce_sum(svd_loss_masked) / self.n_tokens

    # 2-cycles loss
    cycle2_loss = tf.multiply(adj, tf.transpose(adj, [0, 2, 1]))
    i1, i2 = tf.meshgrid(tf.range(batch_size), tf.range(bucket_size), indexing="ij")
    idx = tf.stack([i1, i2, targets3D], axis=-1)
    # targets_mask = tf.scatter_nd(idx, tf.ones([bucket_size, bucket_size]), [batch_size, bucket_size, bucket_size])
    # mask padding and also the correct edges, so this loss doesn't apply to correct predictions
    cycle2_loss_masked = cycle2_loss * self.tokens_to_keep3D # * targets_mask
    cycle2_loss_avg = tf.reduce_sum(cycle2_loss_masked) / self.n_tokens

    # normal log loss
    predictions1D = tf.to_int32(tf.argmax(logits2D, 1))
    probabilities2D = tf.nn.softmax(logits2D)
    cross_entropy1D = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits2D, labels=targets1D)

    correct1D = tf.to_float(tf.equal(predictions1D, targets1D))
    n_correct = tf.reduce_sum(correct1D * tokens_to_keep1D)
    accuracy = n_correct / self.n_tokens
    log_loss = tf.reduce_sum(cross_entropy1D * tokens_to_keep1D) / self.n_tokens

    loss = log_loss + cycle2_loss_avg + svd_loss_avg

    output = {
      'probabilities': tf.reshape(probabilities2D, original_shape),
      'predictions': tf.reshape(predictions1D, flat_shape),
      'tokens': tokens_to_keep1D,
      'correct': correct1D * tokens_to_keep1D,
      'n_correct': n_correct,
      'n_tokens': self.n_tokens,
      'accuracy': accuracy,
      'loss': loss,
      'log_loss': log_loss,
      'svd_loss': svd_loss_avg, #tf.constant(0),
      # 'roots_loss': roots_loss,
      '2cycle_loss': cycle2_loss_avg
    }

    return output
  
  #=============================================================
  def conditional_probabilities(self, logits4D, transpose=True):
    """"""
    
    if transpose:
      logits4D = tf.transpose(logits4D, [0,1,3,2])
    original_shape = tf.shape(logits4D)
    n_classes = original_shape[3]
    
    logits2D = tf.reshape(logits4D, tf.stack([-1, n_classes]))
    probabilities2D = tf.nn.softmax(logits2D)
    return tf.reshape(probabilities2D, original_shape)
  
  #=============================================================
  def tag_argmax(self, tag_probs, tokens_to_keep):
    """"""
    
    return np.argmax(tag_probs[:,Vocab.ROOT:], axis=1)+Vocab.ROOT

  # =============================================================
  def check_cycles_svd(self, parse_preds, length):

    # tokens_to_keep[0] = True
    # length = np.sum(tokens_to_keep)
    # I = np.eye(len(tokens_to_keep))
    # # block loops and pad heads
    # parse_probs = parse_probs * tokens_to_keep * (1 - I)
    # parse_preds = np.argmax(parse_probs, axis=1)
    # tokens = np.arange(1, length)
    # roots = np.where(parse_preds[tokens] == 0)[0] + 1
    #
    laplacian = np.zeros((length - 1, length - 1))
    for i, p in enumerate(parse_preds[1:length]):
      if p != 0:
        laplacian[i, p - 1] = -1.
        laplacian[p - 1, i] = -1.
    degrees = -np.sum(laplacian, axis=0)
    for i, d in enumerate(degrees):
      laplacian[i, i] = d

    e = scipy.linalg.svd(laplacian, compute_uv=False)
    rank = np.sum(np.greater(e, 1e-15))

    adj = np.zeros((len(parse_preds), len(parse_preds)))
    for i, p in enumerate(parse_preds):  # [1:length]):
      if p != 0:
        adj[i, p] = 1

    len_2_cycles = np.sum(np.multiply(adj, np.transpose(adj))) > 0

    # has_cycle = len_2_cycles or (0.5 * np.trace(laplacian) >= rank + 1)
    return int(len_2_cycles), int((0.5 * np.trace(laplacian) >= rank + 1))
  
  #=============================================================
  def parse_argmax(self, parse_probs, tokens_to_keep):
    """"""
    if self.ensure_tree and self.svd_tree:
      tokens_to_keep[0] = True
      length = np.sum(tokens_to_keep)
      I = np.eye(len(tokens_to_keep))
      # block loops and pad heads
      parse_probs = parse_probs * tokens_to_keep * (1 - I)
      parse_preds = np.argmax(parse_probs, axis=1)
      tokens = np.arange(1, length)
      roots = np.where(parse_preds[tokens] == 0)[0] + 1
      roots_lt = 1. if len(roots) < 1 else 0.
      roots_gt = 1. if len(roots) > 1 else 0.
      len_2_cycles, n_cycles = self.check_cycles_svd(parse_preds, length)
      # ensure at least one root
      if roots_lt:
        # The current root probabilities
        root_probs = parse_probs[tokens, 0]
        # The current head probabilities
        old_head_probs = parse_probs[tokens, parse_preds[tokens]]
        # Get new potential root probabilities
        new_root_probs = root_probs / old_head_probs
        # Select the most probable root
        new_root = tokens[np.argmax(new_root_probs)]
        # Make the change
        parse_preds[new_root] = 0
      # ensure at most one root
      elif roots_gt:
        # The probabilities of the current heads
        root_probs = parse_probs[roots, 0]
        # Set the probability of depending on the root zero
        parse_probs[roots, 0] = 0
        # Get new potential heads and their probabilities
        new_heads = np.argmax(parse_probs[roots][:, tokens], axis=1) + 1
        new_head_probs = parse_probs[roots, new_heads] / root_probs
        # Select the most probable root
        new_root = roots[np.argmin(new_head_probs)]
        # Make the change
        parse_preds[roots] = new_heads
        parse_preds[new_root] = 0
      # remove cycles
      if len_2_cycles or n_cycles:
        tarjan = Tarjan(parse_preds, tokens)
        cycles = tarjan.SCCs
        for SCC in tarjan.SCCs:
          if len(SCC) > 1:
            if len(SCC) == 2:
              len_2_cycles = 1.
            else:
              n_cycles = 1.
            dependents = set()
            to_visit = set(SCC)
            while len(to_visit) > 0:
              node = to_visit.pop()
              if not node in dependents:
                dependents.add(node)
                to_visit.update(tarjan.edges[node])
            # The indices of the nodes that participate in the cycle
            cycle = np.array(list(SCC))
            # The probabilities of the current heads
            old_heads = parse_preds[cycle]
            old_head_probs = parse_probs[cycle, old_heads]
            # Set the probability of depending on a non-head to zero
            non_heads = np.array(list(dependents))
            parse_probs[np.repeat(cycle, len(non_heads)), np.repeat([non_heads], len(cycle), axis=0).flatten()] = 0
            # Get new potential heads and their probabilities
            new_heads = np.argmax(parse_probs[cycle][:, tokens], axis=1) + 1
            new_head_probs = parse_probs[cycle, new_heads] / old_head_probs
            # Select the most probable change
            change = np.argmax(new_head_probs)
            changed_cycle = cycle[change]
            old_head = old_heads[change]
            new_head = new_heads[change]
            # Make the change
            parse_preds[changed_cycle] = new_head
            tarjan.edges[new_head].add(changed_cycle)
            tarjan.edges[old_head].remove(changed_cycle)
        return parse_preds, roots_lt, roots_gt, len_2_cycles, n_cycles
    elif self.svd_tree:
      tokens_to_keep[0] = True
      length = np.sum(tokens_to_keep)
      I = np.eye(len(tokens_to_keep))
      # block loops and pad heads
      parse_probs = parse_probs * tokens_to_keep * (1 - I)
      parse_preds = np.argmax(parse_probs, axis=1)
      tokens = np.arange(1, length)
      roots = np.where(parse_preds[tokens] == 0)[0] + 1
      roots_lt = 1. if len(roots) < 1 else 0.
      roots_gt = 1. if len(roots) > 1 else 0.
      len_2_cycles, n_cycles = self.check_cycles_svd(parse_preds, length)
      return parse_preds, roots_lt, roots_gt, len_2_cycles, n_cycles
    elif self.ensure_tree:
      tokens_to_keep[0] = True
      length = np.sum(tokens_to_keep)
      I = np.eye(len(tokens_to_keep))
      # block loops and pad heads
      parse_probs = parse_probs * tokens_to_keep * (1 - I)
      parse_preds = np.argmax(parse_probs, axis=1)
      tokens = np.arange(1, length)
      roots = np.where(parse_preds[tokens] == 0)[0] + 1
      roots_lt = 1. if len(roots) < 1 else 0.
      roots_gt = 1. if len(roots) > 1 else 0.
      # len_2_cycles, n_cycles = self.check_cycles_svd(parse_preds, length)
      # ensure at least one root
      if roots_lt:
        # The current root probabilities
        root_probs = parse_probs[tokens, 0]
        # The current head probabilities
        old_head_probs = parse_probs[tokens, parse_preds[tokens]]
        # Get new potential root probabilities
        new_root_probs = root_probs / old_head_probs
        # Select the most probable root
        new_root = tokens[np.argmax(new_root_probs)]
        # Make the change
        parse_preds[new_root] = 0
      # ensure at most one root
      elif roots_gt:
        # The probabilities of the current heads
        root_probs = parse_probs[roots, 0]
        # Set the probability of depending on the root zero
        parse_probs[roots, 0] = 0
        # Get new potential heads and their probabilities
        new_heads = np.argmax(parse_probs[roots][:, tokens], axis=1) + 1
        new_head_probs = parse_probs[roots, new_heads] / root_probs
        # Select the most probable root
        new_root = roots[np.argmin(new_head_probs)]
        # Make the change
        parse_preds[roots] = new_heads
        parse_preds[new_root] = 0
      # remove cycles
      tarjan = Tarjan(parse_preds, tokens)
      cycles = tarjan.SCCs
      len_2_cycles = 0.
      n_cycles = 0.
      for SCC in tarjan.SCCs:
        if len(SCC) > 1:
          if len(SCC) == 2:
            len_2_cycles = 1.
          else:
            n_cycles = 1.
          dependents = set()
          to_visit = set(SCC)
          while len(to_visit) > 0:
            node = to_visit.pop()
            if not node in dependents:
              dependents.add(node)
              to_visit.update(tarjan.edges[node])
          # The indices of the nodes that participate in the cycle
          cycle = np.array(list(SCC))
          # The probabilities of the current heads
          old_heads = parse_preds[cycle]
          old_head_probs = parse_probs[cycle, old_heads]
          # Set the probability of depending on a non-head to zero
          non_heads = np.array(list(dependents))
          parse_probs[np.repeat(cycle, len(non_heads)), np.repeat([non_heads], len(cycle), axis=0).flatten()] = 0
          # Get new potential heads and their probabilities
          new_heads = np.argmax(parse_probs[cycle][:, tokens], axis=1) + 1
          new_head_probs = parse_probs[cycle, new_heads] / old_head_probs
          # Select the most probable change
          change = np.argmax(new_head_probs)
          changed_cycle = cycle[change]
          old_head = old_heads[change]
          new_head = new_heads[change]
          # Make the change
          parse_preds[changed_cycle] = new_head
          tarjan.edges[new_head].add(changed_cycle)
          tarjan.edges[old_head].remove(changed_cycle)
      return parse_preds, roots_lt, roots_gt, len_2_cycles, n_cycles
    else:
      tokens_to_keep[0] = True
      length = np.sum(tokens_to_keep)
      # block and pad heads
      parse_probs = parse_probs * tokens_to_keep
      parse_preds = np.argmax(parse_probs, axis=1)
      return parse_preds, 0., 0., 0., 0.
    # if self.ensure_tree:
    #   tokens_to_keep[0] = True
    #   length = np.sum(tokens_to_keep)
    #   I = np.eye(len(tokens_to_keep))
    #   # block loops and pad heads
    #   parse_probs = parse_probs * tokens_to_keep * (1-I)
    #   parse_preds = np.argmax(parse_probs, axis=1)
    #   tokens = np.arange(1, length)
    #   roots = np.where(parse_preds[tokens] == 0)[0]+1
    #
    #   # ensure at least one root
    #   roots_lt = False
    #   roots_gt = False
    #   if len(roots) < 1:
    #     roots_lt = True
    #     # The current root probabilities
    #     root_probs = parse_probs[tokens,0]
    #     # The current head probabilities
    #     old_head_probs = parse_probs[tokens, parse_preds[tokens]]
    #     # Get new potential root probabilities
    #     new_root_probs = root_probs / old_head_probs
    #     # Select the most probable root
    #     new_root = tokens[np.argmax(new_root_probs)]
    #     # Make the change
    #     parse_preds[new_root] = 0
    #   # ensure at most one root
    #   elif len(roots) > 1:
    #     roots_gt = True
    #     # # The probabilities of the current heads
    #     # root_probs = parse_probs[roots,0]
    #     # # Set the probability of depending on the root zero
    #     # parse_probs[roots,0] = 0
    #     # # Get new potential heads and their probabilities
    #     # new_heads = np.argmax(parse_probs[roots][:,tokens], axis=1)+1
    #     # new_head_probs = parse_probs[roots, new_heads] / root_probs
    #     # # Select the most probable root
    #     # new_root = roots[np.argmin(new_head_probs)]
    #     # # Make the change
    #     # parse_preds[roots] = new_heads
    #     # parse_preds[new_root] = 0
    #   # remove cycles
    #   tarjan = Tarjan(parse_preds, tokens)
    #   cycles = tarjan.SCCs
    #   tarjan_has_cycle = 0
    #   for SCC in tarjan.SCCs:
    #     if len(SCC) > 1:
    #       tarjan_has_cycle += 1
    #       # dependents = set()
    #       # to_visit = set(SCC)
    #       # while len(to_visit) > 0:
    #       #   node = to_visit.pop()
    #       #   if not node in dependents:
    #       #     dependents.add(node)
    #       #     to_visit.update(tarjan.edges[node])
    #       # # The indices of the nodes that participate in the cycle
    #       # cycle = np.array(list(SCC))
    #       # # The probabilities of the current heads
    #       # old_heads = parse_preds[cycle]
    #       # old_head_probs = parse_probs[cycle, old_heads]
    #       # # Set the probability of depending on a non-head to zero
    #       # non_heads = np.array(list(dependents))
    #       # parse_probs[np.repeat(cycle, len(non_heads)), np.repeat([non_heads], len(cycle), axis=0).flatten()] = 0
    #       # # Get new potential heads and their probabilities
    #       # new_heads = np.argmax(parse_probs[cycle][:,tokens], axis=1)+1
    #       # new_head_probs = parse_probs[cycle, new_heads] / old_head_probs
    #       # # Select the most probable change
    #       # change = np.argmax(new_head_probs)
    #       # changed_cycle = cycle[change]
    #       # old_head = old_heads[change]
    #       # new_head = new_heads[change]
    #       # # Make the change
    #       # parse_preds[changed_cycle] = new_head
    #       # tarjan.edges[new_head].add(changed_cycle)
    #       # tarjan.edges[old_head].remove(changed_cycle)
    #
    #   laplacian = np.zeros((length - 1, length - 1))
    #   for i, p in enumerate(parse_preds[1:length]):
    #     if p != 0:
    #       laplacian[i, p - 1] = -1.
    #       laplacian[p - 1, i] = -1.
    #   degrees = -np.sum(laplacian, axis=0)
    #   for i, d in enumerate(degrees):
    #     laplacian[i, i] = d
    #
    #   e = scipy.linalg.svd(laplacian, compute_uv=False)
    #   rank = np.sum(np.greater(e, 1e-15))
    #
    #   adj = np.zeros((len(parse_preds), len(parse_preds)))
    #   for i, p in enumerate(parse_preds):  # [1:length]):
    #     if p != 0:
    #       adj[i, p] = 1
    #
    #   len_2_cycles = np.sum(np.multiply(adj, np.transpose(adj))) > 0
    #
    #   has_cycle = (0.5 * np.trace(laplacian) >= rank + 1) or len_2_cycles
    #
    #   if has_cycle != (tarjan_has_cycle > 0):
    #
    #     print("Tarjan has cycle: ", tarjan_has_cycle)
    #     print("QR has cycle: ", has_cycle)
    #     print(range(length-1))
    #     print(parse_preds[1:length])
    #     print(parse_preds)
    #     adj = np.zeros((len(parse_preds), len(parse_preds)))
    #     print("adjacency")
    #     # for i, p in enumerate(parse_preds): #[1:length]):
    #     #   if p != 0:
    #     #     # print(i, p)
    #     #     adj[i, p] = 1
    #     for row in adj:
    #       for c in row:
    #         print(str(c) + ", ", end='')
    #       print()
    #
    #     # print("parse_probs", parse_probs)
    #     print("degress", degrees)
    #     print("laplacian", laplacian)
    #     # print("R", R)
    #     print("eig", e)
    #     print("roots_lt", roots_lt)
    #     print("roots_gt", roots_gt)
    #
    #     print("================")
    #
    #   return parse_preds
    # else:
    #   tokens_to_keep[0] = True
    #   length = np.sum(tokens_to_keep)
    #   # block and pad heads
    #   parse_probs = parse_probs * tokens_to_keep
    #   parse_preds = np.argmax(parse_probs, axis=1)
    #   return parse_preds
  
  #=============================================================
  def rel_argmax(self, rel_probs, tokens_to_keep):
    """"""
    
    if self.ensure_tree:
      tokens_to_keep[0] = True
      rel_probs[:,Vocab.PAD] = 0
      root = Vocab.ROOT
      length = np.sum(tokens_to_keep)
      tokens = np.arange(1, length)
      rel_preds = np.argmax(rel_probs, axis=1)
      roots = np.where(rel_preds[tokens] == root)[0]+1
      if len(roots) < 1:
        rel_preds[1+np.argmax(rel_probs[tokens,root])] = root
      elif len(roots) > 1:
        root_probs = rel_probs[roots, root]
        rel_probs[roots, root] = 0
        new_rel_preds = np.argmax(rel_probs[roots], axis=1)
        new_rel_probs = rel_probs[roots, new_rel_preds] / root_probs
        new_root = roots[np.argmin(new_rel_probs)]
        rel_preds[roots] = new_rel_preds
        rel_preds[new_root] = root
      return rel_preds
    else:
      rel_probs[:,Vocab.PAD] = 0
      rel_preds = np.argmax(rel_probs, axis=1)
      return rel_preds
  
  #=============================================================
  def __call__(self, inputs, targets, moving_params=None):
    """"""
    
    raise NotImplementedError()
  
  #=============================================================
  @property
  def global_sigmoid(self):
    return self._global_sigmoid
